# This code references https://huggingface.co/JosephusCheung/ASimilarityCalculatior/blob/main/qwerty.py
# Fill in the path of the model to be queried and the root directory of the reference models, and this script will return the similarity between the model to be queried and all reference models.
import sys, os
import torch
import torch.nn as nn
import torch.nn.functional as F


def cal_cross_attn(to_q, to_k, to_v, rand_input):
    hidden_dim, embed_dim = to_q.shape
    attn_to_q = nn.Linear(hidden_dim, embed_dim, bias=False)
    attn_to_k = nn.Linear(hidden_dim, embed_dim, bias=False)
    attn_to_v = nn.Linear(hidden_dim, embed_dim, bias=False)
    attn_to_q.load_state_dict({"weight": to_q})
    attn_to_k.load_state_dict({"weight": to_k})
    attn_to_v.load_state_dict({"weight": to_v})

    return torch.einsum(
        "ik, jk -> ik",
        F.softmax(
            torch.einsum("ij, kj -> ik", attn_to_q(rand_input), attn_to_k(rand_input)),
            dim=-1,
        ),
        attn_to_v(rand_input),
    )


def model_hash(filename):
    try:
        with open(filename, "rb") as file:
            import hashlib

            m = hashlib.sha256()

            file.seek(0x100000)
            m.update(file.read(0x10000))
            return m.hexdigest()[0:8]
    except FileNotFoundError:
        return "NOFILE"


def eval(model, n, input):
    qk = f"enc_p.encoder.attn_layers.{n}.conv_q.weight"
    uk = f"enc_p.encoder.attn_layers.{n}.conv_k.weight"
    vk = f"enc_p.encoder.attn_layers.{n}.conv_v.weight"
    atoq, atok, atov = model[qk][:, :, 0], model[uk][:, :, 0], model[vk][:, :, 0]

    attn = cal_cross_attn(atoq, atok, atov, input)
    return attn


def main(path, root):
    torch.manual_seed(114514)
    model_a = torch.load(path, map_location="cpu")["weight"]

    print("query:\t\t%s\t%s" % (path, model_hash(path)))

    map_attn_a = {}
    map_rand_input = {}
    for n in range(6):
        hidden_dim, embed_dim, _ = model_a[
            f"enc_p.encoder.attn_layers.{n}.conv_v.weight"
        ].shape
        rand_input = torch.randn([embed_dim, hidden_dim])

        map_attn_a[n] = eval(model_a, n, rand_input)
        map_rand_input[n] = rand_input

    del model_a

    for name in sorted(list(os.listdir(root))):
        path = "%s/%s" % (root, name)
        model_b = torch.load(path, map_location="cpu")["weight"]

        sims = []
        for n in range(6):
            attn_a = map_attn_a[n]
            attn_b = eval(model_b, n, map_rand_input[n])

            sim = torch.mean(torch.cosine_similarity(attn_a, attn_b))
            sims.append(sim)

        print(
            "reference:\t%s\t%s\t%s"
            % (path, model_hash(path), f"{torch.mean(torch.stack(sims)) * 1e2:.2f}%")
        )


if __name__ == "__main__":
    query_path = r"weights\mi v3.pth"
    reference_root = r"weights"
    main(query_path, reference_root)
