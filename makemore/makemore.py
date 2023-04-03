import torch
import torch.nn.functional as F

def main():
    with open("/Users/mish4/nn/makemore/names.txt","r") as f_in:
        words = f_in.read().split()

        chars = sorted(list(set(''.join(words))))
        stoi = {s:i+1 for i,s in enumerate(chars)}
        stoi['.'] = 0
        itos = {i:s for s,i in stoi.items()}

        xs, ys = [], []
        for w in words[:1]:
            chs = ['.'] + list(w) + ['.']
            for ch1, ch2 in zip(chs, chs[1:]):
                ix1 = stoi[ch1] #first letter
                ix2 = stoi[ch2] #following letter
                print(ix1,ix2)
                xs.append(ix1)
                ys.append(ix2)

        xs = torch.tensor(xs)
        ys = torch.tensor(ys)

        print(xs)
        print(ys)

        xenc = F.one_hot(xs, num_classes=27).float()
        print(xenc)
        print(xenc.dtype)

        g = torch.Generator().manual_seed(2147483647)
        W = torch.randn((27,27), generator=g, requires_grad=True) #27 dim input and 27 neurons

        #5x27 27x27 = 5x27
        logits = (xenc @ W)
        counts = logits.exp()
        probs = counts / counts.sum(1, keepdims=True)
        print(probs)

        loss = -probs[torch.arange(5), ys].log().mean()
        print(loss)

        #backward pass
        W.grad = None
        loss.backward()

        W.data += -0.1 * W.grad
if __name__ == "__main__":
    main()
