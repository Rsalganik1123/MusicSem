import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import argparse
from music_encoder import EncoderFactory

class SensitivityAnalyzer:
    def __init__(self, encoder, audio_features, K_list=[1,5,10]):
        self.encoder = encoder
        self.audio_features = audio_features
        self.K_list = sorted(K_list)
        self.max_K = max(K_list)
        
        if audio_features.shape[1] == 1:
            audio_features = audio_features.squeeze(1)
        self.audio_features = audio_features
        print("audio_features.shape:", audio_features.shape:)

        audio_norm = self.audio_features / np.linalg.norm(self.audio_features, axis=1, keepdims=True)
        self.audio_norm = audio_norm.astype(np.float32)
        # print(self.audio_features.shape)

    def get_topk_dict(self, text):
        """Return TopK sets for different K values"""
        try:
            text_feat = self.encoder("text", text)
            text_feat_norm = (text_feat / np.linalg.norm(text_feat)).astype(np.float32)
            
            # print(text_feat_norm.shape)

            sim_scores = np.dot(self.audio_norm, text_feat_norm)
            sorted_indices = np.argsort(-sim_scores)[:self.max_K]
            
            return {
                K: set(sorted_indices[:K]) 
                for K in self.K_list
            }
        except Exception as e:
            print(f"Encoding failed: {str(e)}")
            return None

    def analyze_row(self, original_text, counterfactuals):
        """Analyze a single row of data and return a multi-level results dictionary"""
 
        original_dict = self.get_topk_dict(original_text)
        if not original_dict:
            return None
        
        row_results = {cf: {} for cf in counterfactuals.keys()}
        
        for cf_type, cf_text in counterfactuals.items():
            if pd.isna(cf_text) or not cf_text.strip():
                continue
                
            cf_dict = self.get_topk_dict(cf_text)
            if not cf_dict:
                continue
                
            for K in self.K_list:
                inter = len(original_dict[K] & cf_dict[K])
                row_results[cf_type][K] = inter / K
                
        return row_results




def parse_args():
    parser = argparse.ArgumentParser(
        description="Sensitivity Analysis for Counterfactual Text in Audio Retrieval",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="CLAP",
        choices=["LARP", "CLAP", "imagebind", "clamp3"],
        help="Type of audio-text encoder to use"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the pretrained model checkpoint"
    )
    parser.add_argument(
        "--amodle_of_CLAP",
        type=str,
        default="HTSAT-base",
        choices=['HTSAT-base', 'HTSAT-large', 'HTSAT-tiny', 'HTSAT-tiny-win-1536', 'PANN-6', 'PANN-10', 'PANN-14', 'PANN-14-fmax-8k-20s', 'PANN-14-fmax-18k', 'PANN-14-tiny-transformer', 'PANN-14-win-1536'],
        help="Audio model type for CLAP encoder (ignored for other encoders)"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/all_counterfac.csv",
        help="Path to counterfactuals CSV dataset"
    )
    parser.add_argument(
        "--audio_feat_path",
        type=str,
        required=True,
        help="Path to precomputed audio features (.npy file)"
    )
    parser.add_argument(
        "--K_list",
        nargs="+",
        type=int,
        default=[1, 5, 10],
        help="K values for top-K retrieval metrics"
    )
    parser.add_argument(
        "--cf_types",
        nargs="+",
        type=str,
        default=["Descriptive", "Situational", "Atmospheric", "Metadata", "Contextual"],
        help="Counterfactual types to analyze (must match CSV column names)"
    )
    parser.add_argument(
        "--output_report",
        type=str,
        default="sensitivity_report.json",
        help="Output path for analysis report"
    )
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.encoder == "CLAP":
        encoder = EncoderFactory.create(args.encoder, model_path=args.model_path, autio_model_type=args.amodle_of_CLAP)
    else:
        encoder = EncoderFactory.create(args.encoder, model_path=args.model_path)
    encoder.load_model()

    audio_features = np.load(args.audio_feat_path, allow_pickle=True)
    if audio_features.ndim == 2 and audio_features.shape[1] == 1:
        audio_features = audio_features.squeeze(1)

    analyzer = SensitivityAnalyzer(encoder, audio_features, K_list=args.K_list)    
    df = pd.read_csv(args.data_path).fillna("")
    
    results = {cf: {K: [] for K in args.K_list} for cf in args.cf_types}
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        original_text = row['Original']
        counterfactuals = {cf: row[cf] for cf in args.cf_types}
        
        row_result = analyzer.analyze_row(original_text, counterfactuals)
        if not row_result:
            continue
        
        # Aggregate results
        for cf_type, K_values in row_result.items():
            for K, score in K_values.items():
                results[cf_type][K].append(score)
    
    # Generate final report
    final_report = {}
    for cf_type in args.cf_types:
        final_report[cf_type] = {}
        for K in args.K_list:
            scores = results[cf_type][K]
            if scores:
                final_report[cf_type][f"K={K}"] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'coverage': f"{len(scores)}/{len(df)}"
                }
    
    print("\nSensitivity Analysis Report:")
    print(json.dumps(final_report, indent=2))
    
    with open(args.output_report, 'w') as f:
        json.dump(final_report, f, indent=2)
    print(f"Report saved to {args.output_report}")

if __name__ == '__main__':
    main()

"""
python sensitivity.py \
  --encoder CLAP \
  --model_path /model/tteng/epoch_1.pt \
  --amodle_of_CLAP PANN \
  --audio_feat_path CLAP-sft_MSD-audio_features.npy \
  --data_path data/all_counterfac.csv \
  --output_report  CLAP-sft_MSD-sensitivity.json
"""