# import torch
# from pathlib import Path
# import os


# def path_to_sharedkey(path, dataset_name, classes=None):
#     if dataset_name.lower() == "vggsound":
#         # a generic oneliner which extracts the unique filename for the dataset.
#         # Works on both FakeFolder and VGGSound* datasets
#         sharedkey = Path(path).stem.replace("_mel", "").split("_sample_")[0]
#     elif dataset_name.lower() == "vas":
#         # in the case of vas the procedure is a bit more tricky and involves relying on the premise that
#         # the folder names (.../VAS_validation/cls_0, .../cls_1 etc) are made after enumerating sorted list
#         # of classes.
#         classes = sorted(classes)
#         target_to_label = {f"cls_{i}": c for i, c in enumerate(classes)}
#         # replacing class folder with the name of the class to match the original dataset (cls_2 -> dog)
#         for folder_cls_name, label in target_to_label.items():
#             path = path.replace(folder_cls_name, label).replace(
#                 "melspec_10s_22050hz/", ""
#             )
#         # merging video name with class name to make a unique shared key
#         sharedkey = (
#             Path(path).parent.stem
#             + "_"
#             + Path(path).stem.replace("_mel", "").split("_sample_")[0]
#         )
#     elif dataset_name.lower() == "caps":  # stem : 获取/.之间的部分
#         sharedkey = Path(path).stem.replace("_mel", "").split("_sample_")[0]  # 获得原文件名称
#     else:
#         raise NotImplementedError
#     return sharedkey


# def calculate_kl(featuresdict_1, featuresdict_2, feat_layer_name, same_name=True):
#     # test_input(featuresdict_1, featuresdict_2, feat_layer_name, dataset_name, classes)
#     if not same_name:
#         return (
#             {
#                 "kullback_leibler_divergence_sigmoid": float(-1),
#                 "kullback_leibler_divergence_softmax": float(-1),
#             },
#             None,
#             None,
#         )

#     print(
#         'KL: Assuming that `input2` is "pseudo" target and `input1` is prediction. KL(input2_i||input1_i)'
#     )
#     EPS = 1e-6
#     features_1 = featuresdict_1[feat_layer_name]
#     features_2 = featuresdict_2[feat_layer_name]
#     # # print('features_1 ',features_1.shape) # the predicted (num*10, class_num)
#     # # print('features_2 ',features_2.shape) # the true
#     paths_1 = [os.path.basename(x) for x in featuresdict_1["file_path_"]]
#     paths_2 = [os.path.basename(x) for x in featuresdict_2["file_path_"]]
#     # # print('paths_1 ',len(paths_1)) its path
#     # # print('paths_2 ',len(paths_2))
#     path_to_feats_1 = {p: f for p, f in zip(paths_1, features_1)}
#     # #print(path_to_feats_1)
#     path_to_feats_2 = {p: f for p, f in zip(paths_2, features_2)}
#     # # dataset_name: caps
#     # # in input1 (fakes) can have multiple samples per video, while input2 has only one real
#     # sharedkey_to_feats_1 = {path_to_sharedkey(p, dataset_name, classes): [] for p in paths_1}
#     sharedkey_to_feats_1 = {p: path_to_feats_1[p] for p in paths_1}
#     sharedkey_to_feats_2 = {p: path_to_feats_2[p] for p in paths_2}
#     # sharedkey_to_feats_2 = {path_to_sharedkey(p, dataset_name, classes):path_to_feats_2[p] for p in paths_1}

#     features_1 = []
#     features_2 = []

#     for sharedkey, feat_2 in sharedkey_to_feats_2.items():
#         # print("feat_2",feat_2)
#         if sharedkey not in sharedkey_to_feats_1.keys():
#             print("%s is not in the generation result" % sharedkey)
#             continue
#         features_1.extend([sharedkey_to_feats_1[sharedkey]])
#         # print("feature_step",len(features_1))
#         # print("share",sharedkey_to_feats_1[sharedkey])
#         # just replicating the ground truth logits to compare with multiple samples in prediction
#         # samples_num = len(sharedkey_to_feats_1[sharedkey])
#         features_2.extend([feat_2])

#     features_1 = torch.stack(features_1, dim=0)
#     features_2 = torch.stack(features_2, dim=0)

#     kl_ref = torch.nn.functional.kl_div(
#         (features_1.softmax(dim=1) + EPS).log(),
#         features_2.softmax(dim=1),
#         reduction="none",
#     ) / len(features_1)
#     kl_ref = torch.mean(kl_ref, dim=-1)

#     # AudioGen use this formulation
#     kl_softmax = torch.nn.functional.kl_div(
#         (features_1.softmax(dim=1) + EPS).log(),
#         features_2.softmax(dim=1),
#         reduction="sum",
#     ) / len(features_1)

#     # For multi-class audio clips, this formulation could be better
#     kl_sigmoid = torch.nn.functional.kl_div(
#         (features_1.sigmoid() + EPS).log(), features_2.sigmoid(), reduction="sum"
#     ) / len(features_1)

#     return (
#         {
#             "kullback_leibler_divergence_sigmoid": float(kl_sigmoid),
#             "kullback_leibler_divergence_softmax": float(kl_softmax),
#         },
#         kl_ref,
#         paths_1,
#     )   


# def test_input(featuresdict_1, featuresdict_2, feat_layer_name, dataset_name, classes):
#     assert feat_layer_name == "logits", "This KL div metric is implemented on logits."
#     assert (
#         "file_path_" in featuresdict_1 and "file_path_" in featuresdict_2
#     ), "File paths are missing"
#     assert len(featuresdict_1) >= len(
#         featuresdict_2
#     ), "There are more samples in input1, than in input2"
#     assert (
#         len(featuresdict_1) % len(featuresdict_2) == 0
#     ), "Size of input1 is not a multiple of input1 size."
#     if dataset_name == "vas":
#         assert (
#             classes is not None
#         ), f"Specify classes if you are using vas dataset. Now `classes` – {classes}"
#         print(
#             "KL: when FakesFolder on VAS is used as a dataset, we assume the original labels were sorted",
#             "to produce the target_ids. E.g. `baby` -> `cls_0`; `cough` -> `cls_1`; `dog` -> `cls_2`.",
#         )


# if __name__ == "__main__":
#     # p = torch.tensor([0.25, 0.25, 0.25, 0.25]).view(1, 4)
#     # q = torch.tensor([0.25, 0.25, 0.25, 0.25]).view(1, 4)
#     # 0.

#     p = torch.tensor([0.5, 0.6, 0.7]).view(3, 1)
#     p_ = 1 - p
#     p = torch.cat([p, p_], dim=1).view(-1, 2)
#     print(p)
#     q = torch.tensor([0.5, 0.6, 0.7]).view(3, 1)
#     q_ = 1 - q
#     q = torch.cat([q, q_], dim=1).view(-1, 2)
#     print(q.shape)
#     kl = torch.nn.functional.kl_div(torch.log(q), p, reduction="sum")
#     # 0.0853

#     print(kl)

import torch
from pathlib import Path
import os
import warnings


def _build_sharedkey_map(paths, feats, dataset_name=None, classes=None):
    """
    Returns {shared_key: tensor_of_logits}.
    If `dataset_name` is supplied we derive the shared key with `path_to_sharedkey`,
    otherwise we just use the base filename.
    """
    if dataset_name is None:
        return {os.path.basename(p): f for p, f in zip(paths, feats)}
    return {
        path_to_sharedkey(os.path.basename(p), dataset_name, classes): f
        for p, f in zip(paths, feats)
    }


def calculate_kl(
    featuresdict_1,
    featuresdict_2,
    feat_layer_name,
    same_name=True,
    dataset_name=None,
    classes=None,
):
    """
    Compute KL divergence between a *generation* directory (featuresdict_1)
    and a *reference* directory (featuresdict_2).

    Files that exist in only one of the two directories are *silently skipped*;
    a short warning is printed so the user knows what was dropped.
    """
    if not same_name:
        return (
            {
                "kullback_leibler_divergence_sigmoid": float(-1),
                "kullback_leibler_divergence_softmax": float(-1),
            },
            None,
            None,
        )

    EPS = 1e-6
    feats_1_raw = featuresdict_1[feat_layer_name]
    feats_2_raw = featuresdict_2[feat_layer_name]

    paths_1 = featuresdict_1["file_path_"]
    paths_2 = featuresdict_2["file_path_"]

    # -------- build maps keyed by the *shared* name ----------
    map_1 = _build_sharedkey_map(paths_1, feats_1_raw, dataset_name, classes)
    map_2 = _build_sharedkey_map(paths_2, feats_2_raw, dataset_name, classes)

    common_keys = map_1.keys() & map_2.keys()
    if not common_keys:
        raise ValueError(
            "No overlapping files between generation and reference sets."
        )

    # Notify (once) about files that will be ignored
    dropped_ref = map_2.keys() - common_keys
    dropped_gen = map_1.keys() - common_keys
    if dropped_ref:
        warnings.warn(
            f"{len(dropped_ref)} file(s) present only in reference set will be skipped."
        )
    if dropped_gen:
        warnings.warn(
            f"{len(dropped_gen)} file(s) present only in generation set will be skipped."
        )

    # -------- align tensors ----------
    features_1 = torch.stack([map_1[k] for k in common_keys], dim=0)
    features_2 = torch.stack([map_2[k] for k in common_keys], dim=0)

    # -------- KL computations ----------
    kl_ref = torch.nn.functional.kl_div(
        (features_1.softmax(dim=1) + EPS).log(),
        features_2.softmax(dim=1),
        reduction="none",
    ) / len(features_1)
    kl_ref = torch.mean(kl_ref, dim=-1)

    kl_softmax = torch.nn.functional.kl_div(
        (features_1.softmax(dim=1) + EPS).log(),
        features_2.softmax(dim=1),
        reduction="sum",
    ) / len(features_1)

    kl_sigmoid = torch.nn.functional.kl_div(
        (features_1.sigmoid() + EPS).log(), features_2.sigmoid(), reduction="sum"
    ) / len(features_1)

    return (
        {
            "kullback_leibler_divergence_sigmoid": float(kl_sigmoid),
            "kullback_leibler_divergence_softmax": float(kl_softmax),
        },
        kl_ref,
        list(common_keys),
    )


def test_input(featuresdict_1, featuresdict_2, feat_layer_name, dataset_name, classes):
    """
    Lightweight sanity check that *never* stops the run; it only warns.
    """
    if feat_layer_name != "logits":
        warnings.warn("KL div metric expects logits as input")

    len_gen, len_ref = len(featuresdict_1), len(featuresdict_2)
    if len_gen != len_ref:
        warnings.warn(
            f"Generation set has {len_gen} items, reference set has {len_ref}. "
            "Metrics will be computed on the intersection only."
        )

    if dataset_name == "vas" and classes is None:
        warnings.warn(
            "For the VAS dataset you should pass the `classes` list so that "
            "shared keys are computed correctly."
        )
