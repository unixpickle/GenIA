"""Pinned LPIPS calibration weights, cached alongside torch model weights."""
import torch

REVISION = '082bb24f84c091ea94de2867d34c4544f68e0963'
DIGESTS = {('0.0', 'vgg'): 'b9e4236260c3dd988fc79d2a48d645d885afcbb21f9fd595e6744cf7419b582c', ('0.0', 'alex'): '18720f55913d0af89042f13faa7e536a6ce1444a0914e6db9461355ece1e8cd5', ('0.0', 'squeeze'): 'c27abd3a0145541baa50990817df58d3759c3f8154949f42af3b59b4e042d0bf', ('0.1', 'vgg'): 'a78928a0af1e5f0fcb1f3b9e8f8c3a2a5a3de244d830ad5c1feddc79b8432868', ('0.1', 'alex'): 'df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0', ('0.1', 'squeeze'): '4a5350f23600cb79923ce65bb07cbf57dca461329894153e05a1346bd531cf76'}


def load_calibration(version, net):
    net = 'vgg' if net == 'vgg16' else net
    try:
        digest = DIGESTS[(version, net)]
    except KeyError as exc:
        raise ValueError(f'Unsupported LPIPS version/backbone: {version}/{net}') from exc
    url = f'https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/{REVISION}/lpips/weights/v{version}/{net}.pth'
    return torch.hub.load_state_dict_from_url(url, map_location='cpu',
        file_name=f'lpips-v{version}-{net}-{digest}.pth', check_hash=True, weights_only=True)
