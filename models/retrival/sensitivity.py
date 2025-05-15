import numpy as np
import pandas as pd
import json
from tqdm import tqdm
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

        audio_norm = self.audio_features / np.linalg.norm(self.audio_features, axis=1, keepdims=True)
        self.audio_norm = audio_norm.astype(np.float32)
        print(self.audio_features.shape)

    def get_topk_dict(self, text):
        try:
            text_feat = self.encoder("text", text)
            text_feat_norm = (text_feat / np.linalg.norm(text_feat)).astype(np.float32)
            
            print(text_feat_norm.shape)

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

def main():
    encoder = EncoderFactory.create('imagebind')
    encoder.load_model()

    csv_path = "/model/tteng/MusicEnocdeFactory/data/all_counterfac.csv"
    audio_feat_path = "imagebind-musicCaps-audio_features.npy"
    K_list = [1, 5, 10]
    cf_types = ['Descriptive','Situational','Atmospheric','Metadata','Contextual']
    
    audio_features = np.load(audio_feat_path, allow_pickle=True)
    if audio_features.ndim == 2 and audio_features.shape[1] == 1:
        audio_features = audio_features.squeeze(1)
    
    analyzer = SensitivityAnalyzer(encoder, audio_features, K_list=K_list)
    
    df = pd.read_csv(csv_path).fillna("")
    
    results = {
        cf: {K: [] for K in K_list}
        for cf in cf_types
    }
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        original_text = row['Original']
        counterfactuals = {cf: row[cf] for cf in cf_types}
        
        row_result = analyzer.analyze_row(original_text, counterfactuals)
        if not row_result:
            continue
        
        for cf_type, K_values in row_result.items():
            for K, score in K_values.items():
                results[cf_type][K].append(score)
    
    final_report = {}
    for cf_type in cf_types:
        final_report[cf_type] = {}
        for K in K_list:
            scores = results[cf_type][K]
            if scores:
                final_report[cf_type][f"K={K}"] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'coverage': f"{len(scores)}/{len(df)}"
                }
    
    print("\nSensitivity Analysis Report:")
    print(json.dumps(final_report, indent=2))
    
    with open("imagebind_sensitivity_report.json", 'w') as f:
        json.dump(final_report, f, indent=2)

if __name__ == '__main__':
    main()