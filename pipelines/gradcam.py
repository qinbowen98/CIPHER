"""Per-participant ViT maps targeting a unimodal or fused scalar risk."""
from pathlib import Path
import numpy as np
import torch


class ViTGradCAM:
    def __init__(self, layer, mode='signed', method='historical'):
        self.mode = mode
        self.method = method
        self.handle = layer.register_forward_hook(self.capture)

    def capture(self, module, inputs, output):
        self.activation = output
        output.register_hook(self.capture_gradient)

    def capture_gradient(self, gradient):
        self.gradient = gradient

    def map(self):
        if self.method == 'historical':
            value = (self.gradient.mean(dim=1, keepdim=True) * self.activation).sum(-1)[:, 1:]
        else:
            activation, gradient = self.activation[:, 1:], self.gradient[:, 1:]
            if not torch.isfinite(gradient).all() or not gradient.abs().any():
                raise ValueError('Grad-CAM target layer has no finite nonzero patch gradients.')
            value = (gradient.mean(dim=1, keepdim=True) * activation).sum(-1)
        side = int(value.shape[1] ** 0.5)
        if side * side != value.shape[1]:
            raise ValueError('Expected a square ViT patch grid.')
        value = value.reshape(-1, 1, side, side)
        if self.mode == 'positive':
            value = value.relu()
        return value[:, 0]

    def close(self):
        self.handle.remove()


def export_maps(model, encoder, loader, output, device, fused=True, mode='signed', method='historical', endpoint=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model.eval()
    layer = encoder.blocks[-1] if method == 'historical' else encoder.blocks[-1].norm1
    cam = ViTGradCAM(layer, mode, method)
    count = 0
    try:
        for image, snp, soma, ecg, time, event, pids in loader:
            if len(pids) != 1:
                raise ValueError('Per-participant Grad-CAM requires batch size one.')
            model.zero_grad(set_to_none=True)
            image = image.to(device).requires_grad_(True)
            risk = model(image, snp.to(device), soma.to(device), ecg.to(device)) if fused else model(image)
            risk.sum().backward()
            raw = cam.map()[0].detach().cpu().numpy()
            span = float(np.ptp(raw))
            normalized = (raw - raw.min()) / span if span > 0 else np.zeros_like(raw)
            dataset = getattr(loader, 'dataset', None)
            visit = str(dataset.df.iloc[count].get('visit', '')) if hasattr(dataset, 'df') else ''
            np.savez_compressed(output / ('sample_' + str(count).zfill(6) + '.npz'),
                                pid=np.asarray(pids), visit=np.asarray([visit]), attribution=raw, normalized=normalized,
                                method=np.asarray(method), mode=np.asarray(mode),
                                risk=risk.detach().cpu().numpy(), time=time.numpy(), event=event.numpy())
            name = 'gradcam_' + str(count) + '_' + str(pids[0]) + ('_' + endpoint if endpoint else '')
            np.savetxt(output / (name + '.txt'), raw)
            count += 1
    finally:
        cam.close()
    return count
