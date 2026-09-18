import os, json, time, tempfile, shutil, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
PIPELINE_VERSION = 'v48_oof17_weather_power_experts_20260914'
MODEL_VERSION = 'v48_bridge17_v2'
MANIFEST_NAME = 'meta.json'
SELECTED_KIND = 'distilled'
FILES = {'weather.joblib', 'experts.joblib', 'state.npz', 'meta.json'}
_RUNTIME = None

def script_dir():
    d = Path(__file__).resolve().parent
    return d.parent if d.name == 'utils' else d

def model_dir():
    return Path(os.environ.get('PV_V48_MODEL_DIR', script_dir() / 'model/pv_models_v48_weather_bridge_v2'))

def _runtime():
    global _RUNTIME
    if _RUNTIME is None:
        ns = {'__name__': 'v48_runtime'}
        exec(_RUNTIME_SOURCE, ns)
        _RUNTIME = ns
    return _RUNTIME

def find_pv_root(require_train=True):
    roots = ([Path(os.environ['PV_DATA_ROOT'])] if os.environ.get('PV_DATA_ROOT') else []) + [script_dir() / 'datasets/pv_load_forecasting', script_dir().parent / 'datasets/pv_load_forecasting', Path.cwd() / 'datasets/pv_load_forecasting']
    paths = ['train_data/load_data/pv_load_train.csv', 'train_data/weather_data/pv_weather_train_hourly.csv'] if require_train else ['test_data/weather_data/pv_weather_test_hourly.csv']
    for root in roots:
        if all(((root / q).is_file() for q in paths)):
            return root
    raise FileNotFoundError('V48 missing PV input ' + str(paths))

def find_actual(root):
    explicit = os.environ.get('PV_TA_WEATHER_CSV') or os.environ.get('TA_WEATHER_CSV')
    if explicit:
        if not Path(explicit).is_file():
            raise FileNotFoundError(explicit)
        return Path(explicit)
    paths = [root.parent / 'ta_load_forecasting/train_data/weather_data/tq_weather_data.csv']
    for base in [script_dir(), script_dir().parent, Path.cwd()]:
        paths.extend([base / 'datasets/ta_load_forecasting/train_data/weather_data/tq_weather_data.csv', base / 'data/ta_load_forecasting/train_data/weather_data/tq_weather_data.csv'])
    for q in paths:
        if q.is_file():
            return q
    raise FileNotFoundError('V48 training needs original tq_weather_data.csv. Set PV_TA_WEATHER_CSV; training uses dates before 2025-10-01 only.')

def _validate(meta, state, bridge, experts):
    if meta.get('pipeline') != PIPELINE_VERSION or meta.get('version') != MODEL_VERSION or meta.get('kind') != SELECTED_KIND:
        raise ValueError('Wrong V48 model identity')
    if meta.get('train_end') != '2025-09-30' or meta.get('policy') != 'forecast_only' or str(state['origin']) != '2025-10-01' or (bridge.get('fit_end') != '2025-09-30'):
        raise ValueError('Wrong V48 time boundary')
    shapes = {'pvs': (100,), 'ta': (100,), 'scale': (100,), 'raw_mean': (42,), 'raw_std': (42,), 'weather_mean': (17,), 'weather_std': (17,), 'envelope': (9, 24, 9), 'context': (100, 12), 'origin': ()}
    if set(state) != set(shapes):
        raise ValueError('Wrong V48 state schema')
    for key, shape in shapes.items():
        if state[key].shape != shape:
            raise ValueError('Wrong V48 state shape ' + key)
        if key not in ['pvs', 'origin'] and (not np.isfinite(state[key]).all()):
            raise ValueError('Nonfinite V48 state ' + key)
    if list(state['pvs']) != sorted(('pv_' + str(i) for i in range(1, 101))) or not np.issubdtype(state['ta'].dtype, np.integer) or np.any((state['ta'] < 0) | (state['ta'] > 8)):
        raise ValueError('Wrong V48 users/regions')
    if any((np.any(state[k] <= 0) for k in ['scale', 'raw_std', 'weather_std'])):
        raise ValueError('Invalid V48 scales')
    if bridge['envelope'].shape != (9, 24, 9) or not np.isfinite(bridge['envelope']).all() or bridge['model'].n_features_in_ != 51 or (bridge['model'].n_outputs_ != 17):
        raise ValueError('Invalid V48 weather model')
    if set(bridge['mapping']) != set(('ta_' + str(i) for i in range(1, 10))) or any((v['fit_end'] != '2025-09-30' for v in bridge['mapping'].values())):
        raise ValueError('Invalid V48 station mapping')
    needed = {'pooled_tree', 'local_tree'}
    if set(experts) != needed:
        raise ValueError('Wrong V48 power experts')
    for kind, model in experts.items():
        seq = model if isinstance(model, list) else [model]
        if kind == 'local_tree' and len(seq) != 100:
            raise ValueError('Wrong V48 local expert count')
        for m in seq:
            if m.n_features_in_ != 169:
                raise ValueError('Wrong V48 expert features')
            for tree in m.estimators_:
                if not np.isfinite(tree.tree_.value).all():
                    raise ValueError('Nonfinite V48 tree values')

def _load():
    dest = model_dir()
    if not dest.is_dir() or {q.name for q in dest.iterdir()} != FILES:
        raise FileNotFoundError('V48 needs four complete files; run train_pv.py: ' + str(dest))
    meta = json.loads((dest / 'meta.json').read_text())
    if set(meta.get('sha256', {})) != FILES - {'meta.json'}:
        raise ValueError('Missing V48 checksums')
    for name, digest in meta['sha256'].items():
        if hashlib.sha256((dest / name).read_bytes()).hexdigest() != digest:
            raise ValueError('V48 checksum mismatch ' + name)
    with np.load(dest / 'state.npz', allow_pickle=False) as z:
        state = {k: z[k].copy() for k in z.files}
    bridge = joblib.load(dest / 'weather.joblib')
    experts = joblib.load(dest / 'experts.joblib')
    _validate(meta, state, bridge, experts)
    return (experts, meta, state, bridge)

def train_pv_model():
    rt = _runtime()
    dest = model_dir()
    if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
        if not dest.is_dir() or {q.name for q in dest.iterdir()} != FILES:
            raise RuntimeError('Refuse to overwrite incomplete/foreign V48 directory: ' + str(dest))
        _load()
        print('[PV V48 reuse] validated; no CSV read or fitting', flush=True)
        return dest / 'meta.json'
    begin = time.time()
    budget = float(os.environ.get('PV_V48_MAX_TRAIN_SECONDS', '3600'))
    stage = None
    if budget <= 0:
        raise ValueError('Positive V48 training budget required')

    def check():
        if time.time() - begin > budget:
            raise TimeoutError('V48 training budget exceeded between fitting stages')
    try:
        root = find_pv_root(True)
        d, obs = rt['load_training'](root, find_actual(root), rt['observed17'])
        check()
        oof = rt['make_oof17'](d['a'], obs, d['dates'])
        check()
        k = len(d['dates'])
        state = rt['state17'](d, oof, k)
        rt['D'] = d
        rt['O'] = oof
        rt['OBS'] = obs
        rt['teacher'] = rt['make_teachers'](d, oof, obs)
        check()
        x, y, t, aux, counts = rt['student_data'](k, state)
        check()
        state['context'] = rt['context'](d['y'], state['scale'], k)
        dest.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix='v48_build_', dir=str(dest.parent)))
        experts = rt['fit_student'](x, 0.5 * y + 0.5 * t)
        check()
        bridge, _ = rt['fit_bridge17'](d['a'], obs, d['dates'], k)
        bridge['uncertainty_multiplier'] = rt['uncertainty17'](oof['target'], oof['mean'], oof['base_sd'])
        check()
        joblib.dump(bridge, stage / 'weather.joblib', compress=3)
        joblib.dump(experts, stage / 'experts.joblib', compress=3)
        np.savez_compressed(stage / 'state.npz', **state)
        meta = dict(pipeline=PIPELINE_VERSION, version=MODEL_VERSION, kind=SELECTED_KIND, train_end='2025-09-30', policy='forecast_only', weather_oof=oof['audit'], weather_selection='Forward teacher distillation; no October actual selection', teacher_rows=counts[1], teacher_origins=sorted(rt['teacher']), features=169, device='cpu', seconds=time.time() - begin, training_samples=len(y), legacy_models=False, fields=rt['FIELDS'])
        meta['sha256'] = {name: hashlib.sha256((stage / name).read_bytes()).hexdigest() for name in sorted(FILES - {'meta.json'})}
        _validate(meta, state, bridge, experts)
        (stage / 'meta.json').write_text(json.dumps(meta, indent=2))
        if dest.exists():
            dest.rmdir()
        stage.rename(dest)
        print('[PV V48 trained]', dest, 'seconds=', round(time.time() - begin, 2), flush=True)
        return dest / 'meta.json'
    finally:
        if stage is not None and stage.exists():
            shutil.rmtree(stage)

def predict_pv_model(save_output=True, mode=None):
    if mode not in [None, 'weather_bridge']:
        raise ValueError('V48 only supports weather_bridge mode')
    rt = _runtime()
    experts, meta, state, bridge = _load()
    dates = pd.date_range('2025-10-01', '2025-10-31')
    a = rt['read_hours'](find_pv_root(False) / 'test_data/weather_data/pv_weather_test_hourly.csv', dates)
    mu, sd = rt['bridge_predict17'](bridge, a, dates)
    x = rt['trajectory'](rt['expert_inputs'](state, a, dates, mu, sd, state['context']), 31)
    pred = rt['student_predict'](experts, x, state, 31)
    result = pd.DataFrame({'pv_id': np.repeat(state['pvs'], 31), 'date': np.tile(dates.strftime('%Y/%m/%d'), 100), 'pred': pred.ravel()})
    if len(result) != 3100 or result.duplicated(['pv_id', 'date']).any() or (not np.isfinite(result.pred).all()) or (result.pred < 0).any():
        raise ValueError('Invalid V48 output')
    print('[PV V48 final] pipeline_version=' + PIPELINE_VERSION, 'kind=' + SELECTED_KIND, 'standalone=100% old_models=none forecast_only=true rows=3100 ONLINE SCORE PENDING', flush=True)
    if save_output:
        dest = Path(os.environ.get('OUTPUT_DIR', script_dir() / 'output/pv_load_forecasting'))
        dest.mkdir(parents=True, exist_ok=True)
        result.to_csv(dest / 'submit_result.csv', index=False, encoding='utf-8-sig')
    return result
_RUNTIME_SOURCE = '"""V45 data boundaries and chronological weather bridge; no legacy model dependency."""\nimport os,json,time\nfrom pathlib import Path\nimport numpy as np\nimport pandas as pd\nfrom sklearn.ensemble import ExtraTreesRegressor\nWX=[\'TEM\',\'RHU\',\'PRE_15m\',\'SR\',\'TCC\',\'SWDDIR\',\'SWDDIF\',\'VIS\',\'WS\']\nTAS=[\'ta_\'+str(i) for i in range(1,10)]\nSTART=\'2024-10-01\'\n\ndef read_hours(path,dates,tas=TAS):\n h=pd.read_csv(path);h=h.rename(columns={\'valid_datetime\':\'datetime\'}) if \'datetime\' not in h else h\n h[\'datetime\']=pd.to_datetime(h.datetime);h=h[h.datetime.between(dates[0],dates[-1]+pd.Timedelta(hours=23))]\n idx=pd.MultiIndex.from_product([tas,pd.date_range(dates[0],dates[-1]+pd.Timedelta(hours=23),freq=\'h\')]);actual=pd.MultiIndex.from_frame(h[[\'ta_id\',\'datetime\']])\n if h.duplicated([\'ta_id\',\'datetime\']).any() or len(idx.difference(actual)) or len(actual.difference(idx)):raise ValueError(\'Forecast TA/hour coverage incomplete or duplicate\')\n a=h.set_index([\'ta_id\',\'datetime\']).reindex(idx)[WX].to_numpy(float).reshape(len(tas),len(dates),24,9)\n if not np.isfinite(a).all():raise ValueError(\'Nonfinite forecast\')\n return a\n\ndef observed_daily(path):\n parts=[]\n for c in pd.read_csv(path,usecols=[\'DATETIME\',\'TQ_ID\',\'TEM\',\'RHUM\',\'PRECIPITATION\'],chunksize=250000):\n  dt=pd.to_datetime(c.DATETIME);take=dt.between(START,\'2025-09-30 23:00:00\')&(dt.dt.minute==0)&(dt.dt.second==0)\n  q=c.loc[take].copy();q[\'datetime\']=dt[take];q[\'station\']=\'ta:\'+q.TQ_ID.astype(str)\n  for f,lo,hi in [(\'TEM\',-60,65),(\'RHUM\',0,100),(\'PRECIPITATION\',0,9998)]:\n   v=pd.to_numeric(q[f]);q[f]=v.where(np.isfinite(v)&v.between(lo,hi))\n  parts.append(q[[\'station\',\'datetime\',\'TEM\',\'RHUM\',\'PRECIPITATION\']])\n a=pd.concat(parts,ignore_index=True);dup=a.duplicated([\'station\',\'datetime\'],keep=False)\n if dup.any() and a[dup].groupby([\'station\',\'datetime\'])[[\'TEM\',\'RHUM\',\'PRECIPITATION\']].nunique(dropna=False).gt(1).any().any():raise ValueError(\'Conflicting actual station/hour observations\')\n a=a.drop_duplicates([\'station\',\'datetime\']);a[\'date\']=a.datetime.dt.normalize()\n g=a.groupby([\'station\',\'date\']);o=g.agg(temp=(\'TEM\',\'mean\'),rhu=(\'RHUM\',\'mean\'),rain=(\'PRECIPITATION\',\'sum\'))\n for f,raw in [(\'temp\',\'TEM\'),(\'rhu\',\'RHUM\'),(\'rain\',\'PRECIPITATION\')]:o.loc[g[raw].count()<20,f]=np.nan\n return o\n\ndef calendar(dates):\n return np.stack([np.sin(dates.dayofyear*2*np.pi/365.25),np.cos(dates.dayofyear*2*np.pi/365.25),np.sin(dates.dayofweek*2*np.pi/7),np.cos(dates.dayofweek*2*np.pi/7),dates.day.isin([4,5,6,7]).astype(float)],-1)\n\ndef clean_fit(a):\n # A training-only hourly envelope prevents nonphysical test-night radiation from dominating.\n envelope=np.quantile(np.maximum(a,0),.99,axis=1)\n return envelope\n\ndef raw_features(a,envelope,dates):\n q=a.copy();flag=np.zeros(q.shape[:2]+(1,))\n for j in [3,5,6]:\n  night=envelope[:,:,j]<1.0\n  bad=(q[...,j]>np.maximum(envelope[:,None,:,j]*3,1))&night[:,None,:]\n  flag[...,0]+=bad.mean(-1);q[...,j]=np.where(night[:,None,:],0,np.maximum(q[...,j],0))\n q[...,2]=np.log1p(np.maximum(q[...,2],0));q[...,7]=np.log1p(np.maximum(q[...,7],0))\n # Mean, midday mean, range, and intraday standard deviation retain forecast trajectory information.\n z=np.concatenate([q.mean(2),q[:,:,10:16].mean(2),q.max(2)-q.min(2),q.std(2),flag,np.broadcast_to(calendar(dates),(len(TAS),len(dates),5))],-1)\n return z.astype(\'float32\')\n\ndef fc_daily(a,dates):\n x=np.stack([a[...,0].mean(2),a[...,1].mean(2),a[...,2].sum(2)],-1)\n return pd.DataFrame(x.reshape(-1,3),index=pd.MultiIndex.from_product([TAS,dates],names=[\'ta_id\',\'date\']),columns=[\'temp\',\'rhu\',\'rain\'])\n\ndef corr(a,b):\n v=np.isfinite(a)&np.isfinite(b)\n if v.sum()<30:return np.nan\n if np.std(a[v])<1e-8 or np.std(b[v])<1e-8:return 0.\n return float(np.corrcoef(a[v],b[v])[0,1])\n\ndef map_observed(obs,fc,dates,k):\n hist=dates[max(0,k-180):k];out=[];mapping={}\n for ta in TAS:\n  f=fc.loc[ta].reindex(hist);scores=[]\n  for st in sorted(obs.index.get_level_values(0).unique()):\n   o=obs.loc[st].reindex(hist)\n   value=.5*corr(f.temp.diff().to_numpy(),o.temp.diff().to_numpy())+.3*corr((f.rhu-f.rhu.rolling(15,center=True,min_periods=5).mean()).to_numpy(),(o.rhu-o.rhu.rolling(15,center=True,min_periods=5).mean()).to_numpy())+.2*corr(np.log1p(f.rain.clip(lower=0).to_numpy()),np.log1p(o.rain.clip(lower=0).to_numpy()))\n   if np.isfinite(value):scores.append((value,st))\n  if not scores:raise ValueError(\'Insufficient actual/forecast overlap for \'+ta)\n  score,st=sorted(scores,key=lambda t:(-t[0],t[1]))[0];mapping[ta]={\'station\':st,\'similarity\':score,\'fit_end\':str(dates[k-1].date())}\n  out.append(obs.loc[st].reindex(dates)[[\'temp\',\'rhu\',\'rain\']].to_numpy(float))\n y=np.stack(out);y[...,2]=np.log1p(np.maximum(y[...,2],0));return y,mapping\n\ndef fit_bridge(a,obs,dates,k):\n envelope=clean_fit(a[:,:k]);x=raw_features(a,envelope,dates);y,mapping=map_observed(obs,fc_daily(a,dates),dates,k)\n # Regional identity; all normalizers and all fits stop before the forecast origin.\n x=np.concatenate([x,np.broadcast_to(np.eye(9)[:,None,:],(9,len(dates),9))],-1)\n models=[];res=[]\n for j in range(3):\n  xx=x[:,:k].reshape(-1,x.shape[-1]);yy=y[:,:k,j].reshape(-1);v=np.isfinite(yy)\n  m=ExtraTreesRegressor(n_estimators=120,min_samples_leaf=6,max_features=.8,n_jobs=4,random_state=42+j).fit(xx[v],yy[v]);models.append(m)\n  # Training residual provides only a starting scale. Neural NLL calibrates it on OOF errors.\n  res.append(max(float(np.sqrt(np.mean((m.predict(xx[v])-yy[v])**2))),[1.,3.,.25][j]))\n return dict(models=models,envelope=envelope,residual=np.array(res),mapping=mapping,fit_end=str(dates[k-1].date())),y\n\ndef bridge_predict(model,a,dates):\n x=raw_features(a,model[\'envelope\'],dates);x=np.concatenate([x,np.broadcast_to(np.eye(9)[:,None,:],(9,len(dates),9))],-1);xx=x.reshape(-1,x.shape[-1])\n mu=[];sd=[]\n for j,m in enumerate(model[\'models\']):\n  tree=np.stack([t.predict(xx) for t in m.estimators_]);mu.append(tree.mean(0));sd.append(np.sqrt(tree.var(0)+model[\'residual\'][j]**2))\n return np.stack(mu,-1).reshape(9,len(dates),3),np.stack(sd,-1).reshape(9,len(dates),3)\n\ndef make_oof(a,obs,dates):\n mu=np.zeros((9,len(dates),3));sd=np.zeros_like(mu);target=np.full_like(mu,np.nan);available=np.zeros((9,len(dates),1));audit=[];bridges={}\n # First two months are history only. First trainable horizon begins December 1.\n mu[...,0]=a[...,0].mean(2);mu[...,1]=a[...,1].mean(2);mu[...,2]=np.log1p(np.maximum(a[...,2].sum(2),0));sd[:]=[5,15,1]\n for date in pd.date_range(\'2024-12-01\',dates[-1],freq=\'MS\'):\n  k=dates.get_loc(date);end=min(k+date.days_in_month,len(dates));model,y=fit_bridge(a,obs,dates,k)\n  mm,ss=bridge_predict(model,a[:,k:end],dates[k:end]);mu[:,k:end]=mm;sd[:,k:end]=ss;target[:,k:end]=y[:,k:end];available[:,k:end]=1;bridges[str(date.date())]=model\n  audit.append({\'origin\':str(date.date()),\'fit_end\':model[\'fit_end\'],\'pred_end\':str(dates[end-1].date()),\'mapping\':model[\'mapping\']});print(\'OOF weather\',date.date(),flush=True)\n return dict(mu=mu,sd=sd,target=target,available=available,audit=audit,bridges=bridges)\n\ndef load_training(root,actual_path,actual_reader=None):\n dates=pd.date_range(START,\'2025-09-30\');load=pd.read_csv(Path(root)/\'train_data/load_data/pv_load_train.csv\',parse_dates=[\'date\']);load=load[load.date.isin(dates)]\n pvs=sorted(load.pv_id.unique());expected=sorted(\'pv_\'+str(i) for i in range(1,101))\n if pvs!=expected or load.duplicated([\'pv_id\',\'date\']).any() or not load.groupby(\'pv_id\').ta_id.nunique().eq(1).all():raise ValueError(\'Invalid PV identity/mapping\')\n if len(load)!=100*len(dates):raise ValueError(\'Missing PV/day rows\')\n y=load.pivot(index=\'pv_id\',columns=\'date\',values=\'load\').reindex(index=pvs,columns=dates).to_numpy(float)\n if np.isinf(y).any() or np.any(y[np.isfinite(y)]<0):raise ValueError(\'Invalid historical PV load\')\n ta=np.array([TAS.index(t) for t in load.groupby(\'pv_id\').first().reindex(pvs).ta_id]);a=read_hours(Path(root)/\'train_data/weather_data/pv_weather_train_hourly.csv\',dates)\n return dict(y=y,a=a,dates=dates,pvs=np.array(pvs),ta=ta),(actual_reader or observed_daily)(actual_path)\n\n"""V18-compatible 17 daily/diurnal observed targets, forecast-issued multi-output bridge."""\nimport numpy as np\nimport pandas as pd\nfrom sklearn.ensemble import ExtraTreesRegressor\nFIELDS=[\'temp\',\'temp_max\',\'temp_min\',\'temp_range\',\'rhu\',\'rhu_min\',\'rhu_max\',\'rain\',\'temp_day\',\'rhu_day\',\'rain_day\',\'temp_noon\',\'rhu_noon\',\'rain_noon\',\'temp_night\',\'rhu_night\',\'rain_night\']\nRAIN=[7,10,13,16]\n\ndef aggregate17(h,region=\'station\',temp=\'TEM\',rh=\'RHUM\',rain=\'PRECIPITATION\',observed=True):\n h=h.copy();h[\'date\']=h.datetime.dt.normalize();h[\'hour\']=h.datetime.dt.hour;g=h.groupby([region,\'date\'])\n out=g.agg(temp=(temp,\'mean\'),temp_max=(temp,\'max\'),temp_min=(temp,\'min\'),rhu=(rh,\'mean\'),rhu_min=(rh,\'min\'),rhu_max=(rh,\'max\'),rain=(rain,\'sum\'))\n out[\'temp_range\']=out.temp_max-out.temp_min\n if observed:\n  for prefix,raw in [(\'temp\',temp),(\'rhu\',rh),(\'rain\',rain)]:out.loc[g[raw].count()<20,[c for c in out if c.startswith(prefix)]]=np.nan\n for name,mask in [(\'day\',h.hour.between(6,18)),(\'noon\',h.hour.between(10,15)),(\'night\',h.hour.between(0,5))]:\n  q=h[mask].groupby([region,\'date\']).agg(temp=(temp,\'mean\'),rhu=(rh,\'mean\'),rain=(rain,\'sum\'));out=out.join(q.add_suffix(\'_\'+name))\n return out[FIELDS]\n\ndef observed17(path):\n parts=[]\n for c in pd.read_csv(path,usecols=[\'DATETIME\',\'TQ_ID\',\'TEM\',\'RHUM\',\'PRECIPITATION\'],chunksize=250000):\n  dt=pd.to_datetime(c.DATETIME);take=dt.between(START,\'2025-09-30 23:00:00\')&(dt.dt.minute==0)&(dt.dt.second==0);q=c.loc[take].copy();q[\'datetime\']=dt[take];q[\'station\']=\'ta:\'+q.TQ_ID.astype(str)\n  for f,lo,hi in [(\'TEM\',-60,65),(\'RHUM\',0,100),(\'PRECIPITATION\',0,9998)]:\n   v=pd.to_numeric(q[f]);q[f]=v.where(np.isfinite(v)&v.between(lo,hi))\n  parts.append(q[[\'station\',\'datetime\',\'TEM\',\'RHUM\',\'PRECIPITATION\']])\n h=pd.concat(parts,ignore_index=True);dup=h.duplicated([\'station\',\'datetime\'],keep=False)\n if dup.any() and h[dup].groupby([\'station\',\'datetime\'])[[\'TEM\',\'RHUM\',\'PRECIPITATION\']].nunique(dropna=False).gt(1).any().any():raise ValueError(\'Conflicting TA actual observations\')\n return aggregate17(h.drop_duplicates([\'station\',\'datetime\']))\n\ndef forecast17(a,dates):\n h=pd.DataFrame(a.reshape(-1,9),columns=WX);h[\'ta_id\']=np.repeat(TAS,len(dates)*24);h[\'datetime\']=np.tile(pd.date_range(dates[0],periods=len(dates)*24,freq=\'h\'),9)\n out=aggregate17(h,\'ta_id\',\'TEM\',\'RHU\',\'PRE_15m\',False)\n return np.stack([out.loc[t].reindex(dates).to_numpy(float) for t in TAS])\n\ndef transform17(y):\n z=np.array(y,float,copy=True);z[...,RAIN]=np.log1p(np.maximum(z[...,RAIN],0));return z\n\ndef fit_bridge17(a,obs,dates,k):\n env=clean_fit(a[:,:k]);raw=raw_features(a,env,dates);x=np.concatenate([raw,np.broadcast_to(np.eye(9)[:,None,:],(9,len(dates),9))],-1)\n _,mapping=map_observed(obs,fc_daily(a,dates),dates,k);target=transform17(np.stack([obs.loc[mapping[t][\'station\']].reindex(dates)[FIELDS].to_numpy(float) for t in TAS]));yy=target[:,:k].reshape(-1,17);xm=x[:,:k].reshape(-1,51)\n ym=_v48_nanmean(yy,0);ys=np.maximum(_v48_nanstd(yy,0),.1);v=np.isfinite(yy).all(1)\n if v.sum()<100:raise ValueError(\'Insufficient complete 17-field observed weather rows\')\n model=ExtraTreesRegressor(n_estimators=160,min_samples_leaf=3,max_features=.8,n_jobs=4,random_state=42).fit(xm[v],(yy[v]-ym)/ys)\n residual=np.maximum(np.sqrt(np.mean((model.predict(xm[v])*ys+ym-yy[v])**2,0)),ys*.1)\n return dict(model=model,envelope=env,mean=ym,std=ys,residual=residual,mapping=mapping,fit_end=str(dates[k-1].date())),target\n\ndef bridge_predict17(model,a,dates):\n raw=raw_features(a,model[\'envelope\'],dates);x=np.concatenate([raw,np.broadcast_to(np.eye(9)[:,None,:],(9,len(dates),9))],-1).reshape(-1,51)\n values=np.stack([t.predict(x) for t in model[\'model\'].estimators_]);mu=values.mean(0)*model[\'std\']+model[\'mean\'];sd=np.sqrt(values.var(0)*model[\'std\']**2+model[\'residual\']**2)*model.get(\'uncertainty_multiplier\',np.ones(17))\n # Clamp observed-space means and preserve elementary aggregate consistency.\n for j,f in enumerate(FIELDS):\n  if f.startswith(\'rhu\'):mu[:,j]=np.clip(mu[:,j],0,100)\n  elif f.startswith(\'rain\'):mu[:,j]=np.maximum(mu[:,j],0)\n mu[:,1]=np.maximum(mu[:,1],mu[:,0]);mu[:,2]=np.minimum(mu[:,2],mu[:,0]);mu[:,3]=mu[:,1]-mu[:,2];mu[:,5]=np.minimum(mu[:,5],mu[:,4]);mu[:,6]=np.maximum(mu[:,6],mu[:,4])\n for j in [10,13,16]:mu[:,j]=np.minimum(mu[:,j],mu[:,7])\n return mu.reshape(9,len(dates),17),sd.reshape(9,len(dates),17)\n\ndef make_oof17(a,obs,dates):\n targets=np.full((9,len(dates),17),np.nan);means=np.full_like(targets,np.nan);base_sd=np.full_like(targets,np.nan);bridges={};audit=[]\n for date in pd.date_range(\'2024-12-01\',dates[-1],freq=\'MS\'):\n  k=dates.get_loc(date);end=min(k+date.days_in_month,len(dates));model,y=fit_bridge17(a,obs,dates,k);model[\'uncertainty_multiplier\']=uncertainty17(targets[:,:k],means[:,:k],base_sd[:,:k]);mu,sd=bridge_predict17(model,a[:,k:end],dates[k:end]);means[:,k:end]=mu;base_sd[:,k:end]=sd/model[\'uncertainty_multiplier\'];bridges[str(date.date())]=model;targets[:,k:end]=y[:,k:end];audit.append(dict(origin=str(date.date()),fit_end=model[\'fit_end\'],pred_end=str(dates[end-1].date()),mapping=model[\'mapping\']));print(\'17-field OOF\',date.date(),flush=True)\n return dict(target=targets,mean=means,base_sd=base_sd,bridges=bridges,audit=audit)\n\ndef state17(d,oof,k):\n scale=_v48_nanmedian(np.where(d[\'y\'][:,:k]>0,d[\'y\'][:,:k],np.nan),1);scale=np.where(np.isfinite(scale)&(scale>0),scale,1.)\n env=clean_fit(d[\'a\'][:,:k]);raw=raw_features(d[\'a\'][:,:k],env,d[\'dates\'][:k]);rm=raw.mean((0,1));rs=raw.std((0,1));rs=np.where(rs>.01,rs,1.)\n ym=_v48_nanmean(oof[\'target\'][:,:k],(0,1));ys=np.maximum(_v48_nanstd(oof[\'target\'][:,:k],(0,1)),.1)\n return dict(pvs=d[\'pvs\'],ta=d[\'ta\'],scale=scale,raw_mean=rm,raw_std=rs,weather_mean=ym,weather_std=ys,envelope=env,origin=np.asarray(str((d[\'dates\'][k-1]+pd.Timedelta(days=1)).date())))\n\n\ndef uncertainty17(target,mean,sd):\n valid=np.isfinite(target)&np.isfinite(mean)&np.isfinite(sd)\n count=valid.sum((0,1));num=np.where(valid,(target-mean)**2,0).sum((0,1));den=np.where(valid,sd**2,0).sum((0,1))\n return np.where(count>=30,np.clip(np.sqrt(num/np.maximum(den,1e-8)),.5,4),1.)\n\n"""Independent pooled weather-to-power experts with issue-time level features."""\nimport numpy as np\nfrom joblib import Parallel,delayed\nimport pandas as pd\nfrom sklearn.ensemble import ExtraTreesRegressor,HistGradientBoostingRegressor\n\ndef context(y,scale,k):\n v=y[:,max(0,k-62):k]/scale[:,None];v=np.where(np.isfinite(v),v,np.nan)\n parts=[]\n for n in [3,7,14,28,62]:\n  z=v[:,-n:];parts.extend([_v48_nanmean(z,1),_v48_nanstd(z,1)])\n parts.extend([_v48_nanquantile(v,.1,axis=1),_v48_nanquantile(v,.9,axis=1)])\n return np.nan_to_num(np.stack(parts,-1),nan=1,posinf=1).astype(\'float32\')\n\ndef expert_inputs(state,a,dates,mu,sd,ctx,horizon_offset=0):\n raw=raw_features(a,state[\'envelope\'],dates);f=np.concatenate([np.clip((raw-state[\'raw_mean\'])/state[\'raw_std\'],-8,8),(mu-state[\'weather_mean\'])/state[\'weather_std\'],np.clip(sd/state[\'weather_std\'],.025,5)],-1)[state[\'ta\']]\n n=len(dates);c=np.broadcast_to(ctx[:,None,:],(100,n,12));ident=np.broadcast_to(np.stack([np.arange(100),state[\'ta\']],-1)[:,None,:],(100,n,2));lead=np.broadcast_to(np.arange(horizon_offset,horizon_offset+n)[None,:,None]/31,(100,n,1))\n return np.concatenate([f,c,ident,lead],-1).reshape(100*n,-1)\n\ndef expert_train_data(d,oof,k,state):\n xs=[];ys=[];weights=[];dates=d[\'dates\'];y=d[\'y\'];scale=state[\'scale\'];cache={}\n for s in range(7,k-7,14):\n  n=min(31,k-s);keys=[key for key in oof[\'bridges\'] if pd.Timestamp(key)<=dates[s]]\n  if keys:mu,sd=bridge_predict17(oof[\'bridges\'][max(keys)],d[\'a\'][:,s:s+n],dates[s:s+n])\n  else:\n   mu=transform17(forecast17(d[\'a\'][:,s:s+n],dates[s:s+n]));sd=np.broadcast_to(np.array([5,5,5,3,15,15,15,1,5,15,1,5,15,1,5,15,1]),mu.shape)\n  ctx=context(y,scale,s)\n  x=expert_inputs(state,d[\'a\'][:,s:s+n],dates[s:s+n],mu,sd,ctx);yy=y[:,s:s+n]/scale[:,None];v=np.isfinite(yy)&(yy>=0)\n  xs.append(x[v.ravel()]);ys.append(yy[v]);weights.append(np.full(v.sum(),1.))\n return np.concatenate(xs),np.concatenate(ys),np.concatenate(weights)\n\ndef fit_expert(x,y,w,kind):\n if kind==\'local_tree\':\n  def fit_user(u):\n   v=x[:,88]==u\n   return ExtraTreesRegressor(n_estimators=300,min_samples_leaf=3,max_features=.8,n_jobs=1,random_state=42).fit(x[v,:88],np.log(np.maximum(y[v],.001)),sample_weight=w[v])\n  return Parallel(n_jobs=6,prefer=\'threads\')(delayed(fit_user)(u) for u in range(100))\n if kind==\'pooled_tree\':m=ExtraTreesRegressor(n_estimators=300,min_samples_leaf=6,max_features=.8,n_jobs=6,random_state=42)\n elif kind==\'pooled_boost\':m=HistGradientBoostingRegressor(max_iter=220,max_leaf_nodes=23,min_samples_leaf=30,l2_regularization=10,learning_rate=.045,early_stopping=False,categorical_features=[88,89],random_state=42)\n else:raise ValueError(kind)\n m.fit(x,np.log(np.maximum(y,.001)),sample_weight=w);return m\n\ndef expert_predict(m,x,state,n):\n if isinstance(m,list):return np.stack([np.exp(mm.predict(x[x[:,88]==u,:88])) for u,mm in enumerate(m)])*state[\'scale\'][:,None]\n return np.exp(m.predict(x)).reshape(100,n)*state[\'scale\'][:,None]\ndef observed_for(bridge, dates):\n    return transform17(np.stack([OBS.loc[bridge[\'mapping\'][t][\'station\']].reindex(dates)[FIELDS].to_numpy(float) for t in TAS]))\n\ndef train_arrays(k, state):\n    xs = {n: [] for n in [\'corrected\', \'raw\', \'actual\']}\n    ys = []\n    dates = D[\'dates\']\n    scale = state[\'scale\']\n    missing = 0\n    for s in range(7, k - 7, 14):\n        n = min(31, k - s)\n        dd = dates[s:s + n]\n        a = D[\'a\'][:, s:s + n]\n        keys = [key for key in O[\'bridges\'] if pd.Timestamp(key) <= dates[s]]\n        raw = transform17(forecast17(a, dd))\n        if keys:\n            bridge = O[\'bridges\'][max(keys)]\n            mu, sd = bridge_predict17(bridge, a, dd)\n            actual = observed_for(bridge, dd)\n        else:\n            mu = raw\n            sd = np.broadcast_to(np.array([5, 5, 5, 3, 15, 15, 15, 1, 5, 15, 1, 5, 15, 1, 5, 15, 1]), mu.shape)\n            actual = observed_for(O[\'bridges\'][str(dates[k].date())], dd)\n        missing += int((~np.isfinite(actual)).sum())\n        actual = np.where(np.isfinite(actual), actual, mu)\n        ctx = context(D[\'y\'], scale, s)\n        yy = D[\'y\'][:, s:s + n] / scale[:, None]\n        valid = np.isfinite(yy) & (yy >= 0)\n        for name, w in [(\'corrected\', mu), (\'raw\', raw), (\'actual\', actual)]:\n            xs[name].append(expert_inputs(state, a, dd, w, sd, ctx)[valid.ravel()])\n        ys.append(yy[valid])\n    return ({n: np.concatenate(x) for n, x in xs.items()}, np.concatenate(ys), missing)\ndef trajectory(x, n):\n    a = x.reshape(100, n, -1)\n    wx = np.concatenate([a[:, :, :9], a[:, :, 42:59]], -1)\n    prev = np.concatenate([wx[:, :1], wx[:, :-1]], 1)\n    nxt = np.concatenate([wx[:, 1:], wx[:, -1:]], 1)\n    return np.concatenate([a, wx - prev, nxt - wx, (prev + wx + nxt) / 3], -1).reshape(100 * n, -1)\n\ndef student_data(k, state):\n    xs = []\n    ys = []\n    ts = []\n    states = []\n    counts = [0, 0]\n    for s in range(7, k - 7, 14):\n        n = min(31, k - s)\n        dd = D[\'dates\'][s:s + n]\n        keys = [key for key in O[\'bridges\'] if pd.Timestamp(key) <= dd[0]]\n        if keys:\n            mu, sd = bridge_predict17(O[\'bridges\'][max(keys)], D[\'a\'][:, s:s + n], dd)\n        else:\n            mu = transform17(forecast17(D[\'a\'][:, s:s + n], dd))\n            sd = np.broadcast_to(np.array([5, 5, 5, 3, 15, 15, 15, 1, 5, 15, 1, 5, 15, 1, 5, 15, 1]), mu.shape)\n        x = trajectory(expert_inputs(state, D[\'a\'][:, s:s + n], dd, mu, sd, context(D[\'y\'], state[\'scale\'], s)), n)\n        y = D[\'y\'][:, s:s + n] / state[\'scale\'][:, None]\n        t = y.copy()\n        use = np.zeros(n, bool)\n        for j, day in enumerate(dd):\n            key = str(day.replace(day=1).date())\n            if key in teacher and pd.Timestamp(key) <= dd[0]:\n                t[:, j] = teacher[key][:, day.day - 1] / state[\'scale\']\n                use[j] = True\n        lowcut = _v48_nanquantile(np.where(D[\'y\'][:, :s] > 0, D[\'y\'][:, :s], np.nan), 0.2, axis=1) / state[\'scale\']\n        lowcut = np.where(np.isfinite(lowcut), lowcut, 0.3)\n        actual_low = (y <= lowcut[:, None]).astype(float)\n        soft = 1 / (1 + np.exp(np.clip((np.log(np.maximum(t, 0.001)) - np.log(np.maximum(lowcut[:, None], 0.001))) / 0.25, -30, 30)))\n        soft = np.where(use[None, :], 0.5 * actual_low + 0.5 * soft, actual_low)\n        prior = np.concatenate([(D[\'y\'][:, s - 1] / state[\'scale\'] <= lowcut)[:, None], actual_low[:, :-1]], 1)\n        rec = prior * (y > lowcut[:, None] * 1.5)\n        valid = np.isfinite(y) & (y >= 0)\n        xs.append(x[valid.ravel()])\n        ys.append(np.log(np.maximum(y[valid], 0.001)))\n        ts.append(np.log(np.maximum(t[valid], 0.001)))\n        states.append(np.stack([soft, rec], -1)[valid])\n        counts[0] += int(valid.sum())\n        counts[1] += int((valid & use[None, :]).sum())\n    return (np.concatenate(xs), np.concatenate(ys), np.concatenate(ts), np.concatenate(states), counts)\ndef make_teachers(d,oof,obs):\n from concurrent.futures import ThreadPoolExecutor,as_completed\n def build(date):\n  k=d[\'dates\'].get_loc(date);n=pd.Timestamp(date).days_in_month;dd=d[\'dates\'][k:k+n];bridge=oof[\'bridges\'][date];state=state17(d,oof,k)\n  for key,fallback in [(\'weather_mean\',bridge[\'mean\']),(\'weather_std\',bridge[\'std\'])]:state[key]=np.where(np.isfinite(state[key]),state[key],fallback)\n  xs,y,_=train_arrays(k,state);mu,sd=bridge_predict17(bridge,d[\'a\'][:,k:k+n],dd);act=observed_for(bridge,dd);f=expert_inputs(state,d[\'a\'][:,k:k+n],dd,np.where(np.isfinite(act),act,mu),sd,context(d[\'y\'],state[\'scale\'],k));pr=[]\n  for kind in [\'pooled_tree\',\'local_tree\']:\n   m=fit_expert(xs[\'actual\'],y,np.ones(len(y)),kind);pr.append(expert_predict(m,f,state,n));del m\n  print(\'[PV V48 teacher]\',date,flush=True);return date,np.mean(pr,0)\n dates=[str(d.date()) for d in pd.date_range(\'2024-12-01\',\'2025-09-01\',freq=\'MS\')];teachers={}\n with ThreadPoolExecutor(max_workers=3) as pool:\n  for future in as_completed([pool.submit(build,date) for date in dates]):\n   date,pred=future.result();teachers[date]=pred\n return teachers\n\ndef fit_student(x,target):\n pooled=ExtraTreesRegressor(n_estimators=300,min_samples_leaf=6,max_features=.8,n_jobs=6,random_state=42).fit(x,target)\n def fit_user(u):\n  v=x[:,88]==u\n  return ExtraTreesRegressor(n_estimators=300,min_samples_leaf=3,max_features=.8,n_jobs=1,random_state=42).fit(x[v],target[v])\n local=Parallel(n_jobs=6,prefer=\'threads\')(delayed(fit_user)(u) for u in range(100))\n return {\'pooled_tree\':pooled,\'local_tree\':local}\n\ndef student_predict(experts,x,state,n):\n pooled=np.exp(experts[\'pooled_tree\'].predict(x)).reshape(100,n)\n local=np.stack([np.exp(m.predict(x[x[:,88]==u])) for u,m in enumerate(experts[\'local_tree\'])])\n return .5*(pooled+local)*state[\'scale\'][:,None]\n\n\ndef _v48_nan_reduce(a, axis, operation, q=None):\n    a = np.asarray(a)\n    available = np.any(~np.isnan(a), axis=axis, keepdims=True)\n    present = np.squeeze(available, axis=axis)\n    if a.size == 0:\n        return np.full(present.shape, np.nan)\n    # Only empty slices get temporary values. Their result remains NaN so\n    # existing bridge/context/scale fallbacks retain exactly their semantics.\n    safe = np.where(available, a, 0)\n    reducer = getattr(np, \'nan\' + operation)\n    result = reducer(safe, axis=axis) if q is None else reducer(safe, q, axis=axis)\n    return np.where(present, result, np.nan)\n\ndef _v48_nanmean(a, axis=None):\n    return _v48_nan_reduce(a, axis, \'mean\')\n\ndef _v48_nanstd(a, axis=None):\n    return _v48_nan_reduce(a, axis, \'std\')\n\ndef _v48_nanmedian(a, axis=None):\n    return _v48_nan_reduce(a, axis, \'median\')\n\ndef _v48_nanquantile(a, q, axis=None):\n    return _v48_nan_reduce(a, axis, \'quantile\', q)\n'