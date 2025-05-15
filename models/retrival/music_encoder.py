import importlib


class EncoderFactory:
    _registry = {
        'CLAP': ('CLAP.CLAP_encoder', 'CLAPEncoder'),
        'clamp3': ('clamp3.clamp3_encoder', 'CLAMP3Encoder'),
        'imagebind': ('imagebind.ImageBind_encoder', 'ImageBindEncoder'),
        'LARP': ('LARP.LARP_encoder', 'LARPEncoder'),
    }

    @classmethod
    def create(cls, model_type, **kwargs):
        if model_type not in cls._registry:
            raise ValueError(f"Unknown model type: {model_type}")
        
        module_path, class_name = cls._registry[model_type]
        module = importlib.import_module(module_path)
        encoder_class = getattr(module, class_name)
        return encoder_class(**kwargs)

if  __name__ == "__main__":
    encoder = EncoderFactory.create('LARP')
    encoder.load_model()

    audio_emb = encoder('audio', '/data/tteng/MuLM/SMD/1min_audio/0aF9m87P8Tja3NUMv4DfHt.mp3')
    print("audio emb shape:",audio_emb.shape)

    text_emb = encoder("text", "classical piano piece")
    print("text emb shape:",text_emb.shape)