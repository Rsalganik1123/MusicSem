import numpy as np
import json
import os
from tqdm import tqdm
from sklearn.metrics import label_ranking_average_precision_score
from music_encoder import EncoderFactory
import argparse


class FeatureExtractor:
    def __init__(self, encoder, task="musicCaps"):
        self.encoder = encoder
        self.audio_index_map = {} 
        self.audio_features = []   
        self.text_features = []    
        self.labels = []           
        self.task = task

    def process_dataset(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)

        audio_paths = set()
        for entry in tqdm(data, desc='Collecting audio paths'):
            audio_paths.add(entry['wave_path'])

        for path in tqdm(audio_paths, desc='Extracting audio features'):
            try:
                feature = self.encoder("audio", path) 
                self.audio_index_map[path] = len(self.audio_features)
                self.audio_features.append(feature)
            except Exception as e:
                print(f"Error processing {path}: {str(e)}")

        for entry in tqdm(data, desc='Processing text features'):
            text = entry['caption']
            audio_path = entry['wave_path']
            
            text_feat = self.encoder("text", text)
            self.text_features.append(text_feat)
            
            self.labels.append(self.audio_index_map.get(audio_path, -1))

        np.save(os.path.join(args.output_dir, f"{args.task_name}-text_features.npy"), np.array(self.text_features))
        np.save(os.path.join(args.output_dir, f"{args.task_name}-audio_features.npy"), np.array(self.audio_features))
        np.save(os.path.join(args.output_dir, f"{args.task_name}-labels.npy"), np.array(self.labels))

class RetrievalEvaluator:
    def __init__(self, text_features, audio_features, labels):
        """
        :param text_features: (N, D) 
        :param audio_features: (M, D) 
        :param labels: (N,)
        """
        # print(text_features.shape)
        # print(audio_features.shape)
        if text_features.shape[1] == 1:
            text_features = text_features.squeeze(1)
        self.text_features = text_features
        if audio_features.shape[1] == 1:
            audio_features = audio_features.squeeze(1)
        self.audio_features = audio_features
        print("text_features.shape:", text_features.shape)
        print("audio_features.shape:", audio_features.shape)
        self.labels = labels
        self.sim_matrix = None

    def compute_similarity(self):
        text_norm = self.text_features / np.linalg.norm(self.text_features, axis=1, keepdims=True)
        audio_norm = self.audio_features / np.linalg.norm(self.audio_features, axis=1, keepdims=True)
        self.sim_matrix = np.dot(text_norm, audio_norm.T)
        return self.sim_matrix

    def evaluate(self):
        metrics = {
            'R@1': [], 'R@5': [], 'R@10': [],
            'NDCG@5': [], 'NDCG@10': [],
            'MRR': []
        }

        for i in range(len(self.text_features)):
            scores = self.sim_matrix[i]
            sorted_indices = np.argsort(-scores)
            true_idx = self.labels[i]

            # Recall@K
            metrics['R@1'].append(true_idx in sorted_indices[:1])
            metrics['R@5'].append(true_idx in sorted_indices[:5])
            metrics['R@10'].append(true_idx in sorted_indices[:10])

            # NDCG@K
            for k in [5, 10]:
                relevance = [1 if idx == true_idx else 0 for idx in sorted_indices[:k]]
                dcg = sum([rel / np.log2(pos+2) for pos, rel in enumerate(relevance)])
                idcg = sum([1.0 / np.log2(pos+2) for pos in range(min(k, 1))])
                metrics[f'NDCG@{k}'].append(dcg / idcg if idcg > 0 else 0)

            # MRR
            rank = np.where(sorted_indices == true_idx)[0]
            metrics['MRR'].append(1/(rank[0]+1) if rank.size > 0 else 0)


        metrics['R@1'] = np.mean(metrics['R@1'])
        metrics['R@5'] = np.mean(metrics['R@5'])
        metrics['R@10'] = np.mean(metrics['R@10'])
        metrics['NDCG@5'] = np.mean(metrics['NDCG@5'])
        metrics['NDCG@10'] = np.mean(metrics['NDCG@10'])
        metrics['MRR'] = np.mean(metrics['MRR'])
        
        y_true = np.zeros_like(self.sim_matrix)
        for i, idx in enumerate(self.labels):
            if idx != -1:
                y_true[i, idx] = 1
        metrics['MAP'] = label_ranking_average_precision_score(y_true, self.sim_matrix)

        return metrics

import numpy as np

def calculate_random_scores(M):
    if M < 1:
        raise ValueError("M must be at least 1")
    
    scores = {}
    
    scores['R@1'] = min(1, M) / M
    scores['R@5'] = min(5, M) / M
    scores['R@10'] = min(10, M) / M
    
    def compute_ndcg_k(K):
        effective_K = min(K, M)
        sum_dcg = sum(1 / np.log2(pos + 2) for pos in range(effective_K))
        return sum_dcg / M
    
    scores['NDCG@5'] = compute_ndcg_k(5)
    scores['NDCG@10'] = compute_ndcg_k(10)
    
    harmonic_series = sum(1 / r for r in range(1, M + 1))
    scores['MRR'] = harmonic_series / M
    scores['MAP'] = scores['MRR']
    
    return scores

def parse_args():
    parser = argparse.ArgumentParser(description="Cross-Modal Retrieval Evaluation Tool")
    
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/MSD-Eval.json",
        help="Path to the input dataset JSON file"
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="clamp3",
        choices=["LARP", "CLAP", "imagebind", "clamp3"],
        help="Type of encoder to use (default: clamp3)"
    )
    parser.add_argument(
        "--model_path",
        default=None,
        type=str
    )    
    parser.add_argument(
        "--amodle_of_CLAP",
        type=str,
        default="Default",
        choices=["Default", 'HTSAT-base', 'HTSAT-large', 'HTSAT-tiny', 'HTSAT-tiny-win-1536', 'PANN-6', 'PANN-10', 'PANN-14', 'PANN-14-fmax-8k-20s', 'PANN-14-fmax-18k', 'PANN-14-tiny-transformer', 'PANN-14-win-1536'],
        help="Type of audio encoder of CLAP (default: HTSAT-base)"
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="tmp",
        help="Task identifier for naming outputs (default: tmp)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./",
        help="Output directory for features and results (default: ./)"
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.encoder == "CLAP":
        encoder = EncoderFactory.create(args.encoder, model_path=args.model_path, audio_model_type=args.amodle_of_CLAP)
    else:
        encoder = EncoderFactory.create(args.encoder, model_path=args.model_path)
    encoder.load_model()
    
    extractor = FeatureExtractor(encoder, task=args.task_name)
    extractor.process_dataset(args.dataset_path)
    
    text_feats = np.load(os.path.join(args.output_dir, f"{args.task_name}-text_features.npy"), allow_pickle=True)
    audio_feats = np.load(os.path.join(args.output_dir, f"{args.task_name}-audio_features.npy"), allow_pickle=True)
    labels = np.load(os.path.join(args.output_dir, f"{args.task_name}-labels.npy"), allow_pickle=True)
    
    evaluator = RetrievalEvaluator(text_feats, audio_feats, labels)
    evaluator.compute_similarity()
    results = evaluator.evaluate()
    
    output_path = os.path.join(args.output_dir, f"{args.task_name}-result.json")
    with open(output_path, 'w') as f:
        json.dump(results, f)
    
    print("\nEvaluation Results:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
    print(f"\nResults saved to {output_path}")

# For 630k_best:
# python retrieval_eval.py --dataset_path data/MSD-Eval.json --encoder CLAP --amodle_of_CLAP Default --task_name CLAP-sft-111_MSD --model_path /model/tteng/MusicSem/models/retrieval/epoch_111.pt
# For music_audioset_epoch_15_esc_90.14:
# python retrieval_eval.py --dataset_path data/MSD-Eval.json --encoder CLAP --amodle_of_CLAP HTSAT-base --task_name CLAP-sft_MSD --model_path /model/tteng/MusicEnocdeFactory/models/CLAP/ckpt/music_audioset_epoch_15_esc_90.14.pt