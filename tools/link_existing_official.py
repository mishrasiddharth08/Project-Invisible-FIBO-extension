"""Explicitly link existing pinned assets/configs; no network, conversion or install."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from forge_fibo2.provenance import identify,PINS

def runtime_directories(folder):
    folder=Path(folder).resolve()
    result={}
    for role in ('transformer','text_encoder','vae'):
        path=folder/'configs'/role
        if not (path/'config.json').is_file():path=folder/role
        if not (path/'config.json').is_file():raise ValueError(f'Missing local {role}/config.json')
        result[role]={'directory':str(path)}
    for name in ('processor','scheduler','fonts'):
        if not (folder/name).is_dir():raise ValueError(f'Missing local {name} folder')
    for name in ('Aileron-Regular.otf','DejaVuSans.ttf'):
        if not (folder/'fonts'/name).is_file():raise ValueError(f'Missing official font: {name}')
    if not (folder/'model_index.json').is_file():raise ValueError('Missing local model_index.json')
    return {'model_index':str(folder/'model_index.json'),'component_configs':result,
            **{name:str(folder/name) for name in ('processor','scheduler','fonts')}}

def build_manifest(forge_root,runtime_dir,checkpoint,te_bf16,te_int8,vae):
    forge_root=Path(forge_root).resolve()
    if not (forge_root/'modules/processing.py').is_file():raise ValueError('Not a Forge root')
    runtime=runtime_directories(runtime_dir)
    components={};identities={}
    for key,value,role,precision in (
        ('checkpoint',checkpoint,'transformer',None),('text_encoder_bf16',te_bf16,'text_encoder','bf16'),
        ('text_encoder_int8_convrot',te_int8,'text_encoder','int8_convrot'),('vae',vae,'vae','fp32')):
        path=Path(value).resolve();identity=identify(path,role)
        if precision and identity['precision']!=precision:raise ValueError(f'Wrong precision for {key}')
        identities[key]=identity
        components[key]={'path':str(path),'sha256':identity['sha256'],'bytes':path.stat().st_size,'available':True}
    components['text_encoder_preference']=['bf16','int8_convrot']
    runtime['host_root']=str(forge_root)
    return {'schema':1,'family':'fibo2','family_display':'FIBO-2.0',
            'repository':PINS['repository'],'revision':PINS['revision'],
            'variant':identities['checkpoint']['variant'],'components':components,'runtime':runtime}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('forge-root','runtime-dir','checkpoint','te-bf16','te-int8','vae'):
        parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--manifest',type=Path,default=ROOT/'model-manifest.json')
    args=parser.parse_args()
    if args.manifest.exists():raise FileExistsError('Manifest exists; back it up before replacing it.')
    value=build_manifest(args.forge_root,args.runtime_dir,args.checkpoint,args.te_bf16,args.te_int8,args.vae)
    args.manifest.write_text(json.dumps(value,indent=2),encoding='utf-8')
    print('Pinned existing assets linked. No download, conversion, environment change or GPU job ran.')

if __name__=='__main__':main()
