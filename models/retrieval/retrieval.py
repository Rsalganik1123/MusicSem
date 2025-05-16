import argparse
import numpy as np
import json
import os
from tqdm import tqdm
from music_encoder import EncoderFactory

class FeatureExtractor:
    """Feature extractor for audio-text dataset with auto-saving capabilities"""
    def __init__(self, encoder, feature_dir):
        """
        :param encoder: Pretrained encoder model
        :param feature_dir: Directory to store/load features
        """
        self.encoder = encoder
        self.feature_dir = feature_dir
        self.audio_index_map = {}   # {audio_path: index}
        self.audio_features = []    # [audio_feature]
        self.text_features = []     # [text_feature]
        self.labels = []            # Matching indices
        self.audio_paths = []       # List of unique audio paths
        self.texts = []             # List of all text captions
        os.makedirs(feature_dir, exist_ok=True)

    def process_dataset(self, json_path):
        """Process dataset and save features to disk"""
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Collect unique audio paths and all text captions
        audio_paths_set = set()
        self.texts = []
        for entry in tqdm(data, desc='Collecting metadata'):
            audio_paths_set.add(entry['wave_path'])
            self.texts.append(entry['caption'])
        self.audio_paths = list(audio_paths_set)

        # Save metadata
        np.save(os.path.join(self.feature_dir, 'audio_paths.npy'), self.audio_paths)
        np.save(os.path.join(self.feature_dir, 'texts.npy'), self.texts)

        # Extract audio features
        self.audio_features = []
        for path in tqdm(self.audio_paths, desc='Processing audio'):
            try:
                feat = self.encoder("audio", path)
                self.audio_index_map[path] = len(self.audio_features)
                self.audio_features.append(feat)
            except Exception as e:
                print(f"Error processing {path}: {str(e)}")

        # Extract text features and create labels
        self.text_features = []
        self.labels = []
        for entry in tqdm(data, desc='Processing text'):
            text_feat = self.encoder("text", entry['caption'])
            self.text_features.append(text_feat)
            audio_idx = self.audio_index_map.get(entry['wave_path'], -1)
            self.labels.append(audio_idx)

        # Save features
        np.save(os.path.join(self.feature_dir, 'audio_features.npy'), np.array(self.audio_features))
        np.save(os.path.join(self.feature_dir, 'text_features.npy'), np.array(self.text_features))
        np.save(os.path.join(self.feature_dir, 'labels.npy'), np.array(self.labels))

class CrossModalRetriever:
    """Cross-modal retriever for text<->audio search"""
    def __init__(self, audio_features, text_features, audio_paths, texts):
        """
        :param audio_features: (N, D) numpy array
        :param text_features: (M, D) numpy array
        :param audio_paths: List[N] of audio paths
        :param texts: List[M] of text captions
        """
        # L2 normalize features for cosine similarity
        self.audio_features = audio_features / np.linalg.norm(audio_features, axis=1, keepdims=True)
        self.text_features = text_features / np.linalg.norm(text_features, axis=1, keepdims=True)
        self.audio_paths = audio_paths
        self.texts = texts

    def search(self, query_feature, modality='text', top_k=10):
        """
        Perform cross-modal search
        :param query_feature: Input feature vector (D,)
        :param modality: Query type: 'text' or 'audio'
        :param top_k: Number of results to return
        :return: Tuple of (indices, scores, results)
        """
        # Normalize query
        query_norm = query_feature / np.linalg.norm(query_feature)
        
        # Determine search direction
        if modality == 'text':
            target_features = self.audio_features
            result_data = self.audio_paths
        elif modality == 'audio':
            target_features = self.text_features
            result_data = self.texts
        else:
            raise ValueError(f"Invalid modality: {modality}")

        # Compute similarity scores
        scores = np.dot(target_features, query_norm)
        top_indices = np.argsort(-scores)[:top_k]
        
        return top_indices, scores[top_indices], [result_data[i] for i in top_indices]

def load_features(feature_dir):
    """Load precomputed features from directory"""
    required_files = ['audio_features.npy', 'text_features.npy', 
                     'audio_paths.npy', 'texts.npy']
    for f in required_files:
        if not os.path.exists(os.path.join(feature_dir, f)):
            raise FileNotFoundError(f"Missing feature file: {f}")
    
    return (
        np.load(os.path.join(feature_dir, 'audio_features.npy')),
        np.load(os.path.join(feature_dir, 'text_features.npy')),
        np.load(os.path.join(feature_dir, 'audio_paths.npy'), allow_pickle=True).tolist(),
        np.load(os.path.join(feature_dir, 'texts.npy'), allow_pickle=True).tolist()
    )

def main():
    parser = argparse.ArgumentParser(description='Cross-modal audio-text retrieval system')
    parser.add_argument('--query', required=True, 
                       help='Input text or path to audio file')
    parser.add_argument('--feature_dir', default='./',
                       help='Directory containing/pre-storing features')
    parser.add_argument('--dataset_json', default="data/MSD-Eval.json", 
                       help='Dataset JSON path for feature extraction')
    parser.add_argument('--top_k', type=int, default=10,
                       help='Number of results to return')
    parser.add_argument("--encoder", type=str, default="clamp3",
                        choices=["LARP", "CLAP", "imagebind", "clamp3"],
                        help="Type of encoder to use (default: clamp3)"
    )
    args = parser.parse_args()

    # Auto-create features if missing
    if not os.path.exists(os.path.join(args.feature_dir, 'audio_features.npy')):
        if not args.dataset_json:
            raise ValueError("Features not found and dataset_json not provided")
        
        print("Generating features...")
        encoder = EncoderFactory.create(args.encoder)
        encoder.load_model()
        extractor = FeatureExtractor(encoder, args.feature_dir)
        extractor.process_dataset(args.dataset_json)

    # Load precomputed features
    audio_feats, text_feats, audio_paths, texts = load_features(args.feature_dir)

    # Create encoder and process query
    encoder = EncoderFactory.create(args.encoder)
    encoder.load_model()
    
    # Determine query type
    if os.path.isfile(args.query) and args.query.lower().endswith(('.wav', '.mp3')):
        modality = 'audio'
        try:
            query_feat = encoder("audio", args.query)
        except Exception as e:
            print(f"Audio processing error: {str(e)}")
            return
    else:
        modality = 'text'
        query_feat = encoder("text", args.query)

    # Perform search
    retriever = CrossModalRetriever(audio_feats, text_feats, audio_paths, texts)
    indices, scores, results = retriever.search(query_feat, modality, args.top_k)

    # Display results
    print(f"\nTop {args.top_k} {modality} query results:")
    for i, (score, result) in enumerate(zip(scores, results)):
        print(f"{i+1}. [{score:.4f}] {result}")

if __name__ == '__main__':
    main()