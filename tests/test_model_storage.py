import os
import subprocess
import sys


def test_cache_root_dry_run(tmp_path):
    root = tmp_path / "external models"
    env = os.environ.copy()
    # Verify explicit --cache-root wins without changing where login is stored.
    env["HF_HUB_CACHE"] = str(tmp_path / "wrong-hub")
    env["TORCH_HOME"] = str(tmp_path / "wrong-torch")
    result = subprocess.run([sys.executable, "-m", "genia.core.download_weights",
        "--dry-run", "--no-actionmesh", "--cache-root", str(root)],
        env=env, capture_output=True, text=True, check=True)
    assert str(root / "huggingface" / "hub") in result.stdout
    assert str(root / "torch" / "hub") in result.stdout
    assert "facebook/sam-3d-objects" in result.stdout
    assert "facebook/map-anything" in result.stdout
    assert "Ruicheng/moge-vitl" in result.stdout
    assert "ActionMesh" not in result.stdout
    assert not root.exists()


def test_download_resumes_interrupted_stream(monkeypatch):
    import huggingface_hub
    import time
    from requests.exceptions import ChunkedEncodingError
    from genia.core.download_weights import fetch_hf
    calls=[]
    def download(**kwargs):
        calls.append(kwargs)
        if len(calls)==1: raise ChunkedEncodingError('interrupted body')
        return '/cached/model'
    monkeypatch.setattr(huggingface_hub,'snapshot_download',download)
    monkeypatch.setattr(time,'sleep',lambda _:None)
    assert fetch_hf('test/model',False,['*.pt'])==0
    assert len(calls)==2 and calls[0]==calls[1]
    assert calls[0]['max_workers']==1


def test_download_does_not_retry_bad_requests(monkeypatch):
    import huggingface_hub
    import requests
    import pytest
    from genia.core.download_weights import fetch_hf
    calls=[]
    def download(**kwargs):
        calls.append(kwargs)
        response=requests.Response(); response.status_code=404
        raise requests.HTTPError('not found',response=response)
    monkeypatch.setattr(huggingface_hub,'snapshot_download',download)
    with pytest.raises(requests.HTTPError): fetch_hf('test/model',False)
    assert len(calls)==1
