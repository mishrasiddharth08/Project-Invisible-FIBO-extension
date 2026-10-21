from pathlib import Path
from types import SimpleNamespace
import importlib.util
import pytest

path=Path(__file__).resolve().parents[1]/'tools/link_existing_official.py'
spec=importlib.util.spec_from_file_location('link_existing_official',path)
tool=importlib.util.module_from_spec(spec);spec.loader.exec_module(tool)

def configuration(tmp_path):
    folder=tmp_path/'runtime';folder.mkdir()
    (folder/'model_index.json').write_text('{}')
    for role in ('transformer','text_encoder','vae'):
        directory=folder/'configs'/role;directory.mkdir(parents=True)
        (directory/'config.json').write_text('{}')
    for name in ('processor','scheduler','fonts'):(folder/name).mkdir()
    for name in ('Aileron-Regular.otf','DejaVuSans.ttf'):(folder/'fonts'/name).write_bytes(b'font fixture')
    return folder

def test_configuration_only_bundle_and_full_snapshot_layouts(tmp_path):
    folder=configuration(tmp_path)
    assert tool.runtime_directories(folder)['component_configs']['vae']['directory']==str(folder/'configs/vae')
    (folder/'configs/vae').rename(folder/'vae')
    assert tool.runtime_directories(folder)['component_configs']['vae']['directory']==str(folder/'vae')

def test_build_manifest_preserves_verified_variant_and_precisions(tmp_path,monkeypatch):
    folder=configuration(tmp_path)
    host=tmp_path/'forge';(host/'modules').mkdir(parents=True);(host/'modules/processing.py').write_text('')
    files=[]
    for name in ('checkpoint','bf16','int8','vae'):
        file=tmp_path/(name+'.safetensors');file.write_bytes(b'fixture');files.append(file)
    def identify(path,role):
        precision={'checkpoint':'int8_convrot','bf16':'bf16','int8':'int8_convrot','vae':'fp32'}[path.stem]
        return {'precision':precision,'variant':'dev','sha256':'0'*64}
    monkeypatch.setattr(tool,'identify',identify)
    value=tool.build_manifest(host,folder,*files)
    assert value['variant']=='dev' and value['repository']==tool.PINS['repository']
    assert value['components']['text_encoder_preference']==['bf16','int8_convrot']
    assert not (tmp_path/'model-manifest.json').exists()

def test_missing_local_resources_fail_before_manifest_creation(tmp_path):
    folder=configuration(tmp_path);(folder/'fonts/DejaVuSans.ttf').unlink()
    with pytest.raises(ValueError,match='font'):tool.runtime_directories(folder)
