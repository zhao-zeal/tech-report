"""V45 data boundaries and chronological weather bridge; no legacy model dependency."""
import os,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
WX=['TEM','RHU','PRE_15m','SR','TCC','SWDDIR','SWDDIF','VIS','WS']
TAS=['ta_'+str(i) for i in range(1,10)]
START='2024-10-01'

def read_hours(path,dates,tas=TAS):
 h=pd.read_csv(path);h=h.rename(columns={'valid_datetime':'datetime'}) if 'datetime' not in h else h
 h['datetime']=pd.to_datetime(h.datetime);h=h[h.datetime.between(dates[0],dates[-1]+pd.Timedelta(hours=23))]
 idx=pd.MultiIndex.from_product([tas,pd.date_range(dates[0],dates[-1]+pd.Timedelta(hours=23),freq='h')]);actual=pd.MultiIndex.from_frame(h[['ta_id','datetime']])
 if h.duplicated(['ta_id','datetime']).any() or len(idx.difference(actual)) or len(actual.difference(idx)):raise ValueError('Forecast TA/hour coverage incomplete or duplicate')
 a=h.set_index(['ta_id','datetime']).reindex(idx)[WX].to_numpy(float).reshape(len(tas),len(dates),24,9)
 if not np.isfinite(a).all():raise ValueError('Nonfinite forecast')
 return a

def observed_daily(path):
 parts=[]
 for c in pd.read_csv(path,usecols=['DATETIME','TQ_ID','TEM','RHUM','PRECIPITATION'],chunksize=250000):
  dt=pd.to_datetime(c.DATETIME);take=dt.between(START,'2025-09-30 23:00:00')&(dt.dt.minute==0)&(dt.dt.second==0)
  q=c.loc[take].copy();q['datetime']=dt[take];q['station']='ta:'+q.TQ_ID.astype(str)
  for f,lo,hi in [('TEM',-60,65),('RHUM',0,100),('PRECIPITATION',0,9998)]:
   v=pd.to_numeric(q[f]);q[f]=v.where(np.isfinite(v)&v.between(lo,hi))
  parts.append(q[['station','datetime','TEM','RHUM','PRECIPITATION']])
 a=pd.concat(parts,ignore_index=True);dup=a.duplicated(['station','datetime'],keep=False)
 if dup.any() and a[dup].groupby(['station','datetime'])[['TEM','RHUM','PRECIPITATION']].nunique(dropna=False).gt(1).any().any():raise ValueError('Conflicting actual station/hour observations')
 a=a.drop_duplicates(['station','datetime']);a['date']=a.datetime.dt.normalize()
 g=a.groupby(['station','date']);o=g.agg(temp=('TEM','mean'),rhu=('RHUM','mean'),rain=('PRECIPITATION','sum'))
 for f,raw in [('temp','TEM'),('rhu','RHUM'),('rain','PRECIPITATION')]:o.loc[g[raw].count()<20,f]=np.nan
 return o

def calendar(dates):
 return np.stack([np.sin(dates.dayofyear*2*np.pi/365.25),np.cos(dates.dayofyear*2*np.pi/365.25),np.sin(dates.dayofweek*2*np.pi/7),np.cos(dates.dayofweek*2*np.pi/7),dates.day.isin([4,5,6,7]).astype(float)],-1)

def clean_fit(a):
 # A training-only hourly envelope prevents nonphysical test-night radiation from dominating.
 envelope=np.quantile(np.maximum(a,0),.99,axis=1)
 return envelope

def raw_features(a,envelope,dates):
 q=a.copy();flag=np.zeros(q.shape[:2]+(1,))
 for j in [3,5,6]:
  night=envelope[:,:,j]<1.0
  bad=(q[...,j]>np.maximum(envelope[:,None,:,j]*3,1))&night[:,None,:]
  flag[...,0]+=bad.mean(-1);q[...,j]=np.where(night[:,None,:],0,np.maximum(q[...,j],0))
 q[...,2]=np.log1p(np.maximum(q[...,2],0));q[...,7]=np.log1p(np.maximum(q[...,7],0))
 # Mean, midday mean, range, and intraday standard deviation retain forecast trajectory information.
 z=np.concatenate([q.mean(2),q[:,:,10:16].mean(2),q.max(2)-q.min(2),q.std(2),flag,np.broadcast_to(calendar(dates),(len(TAS),len(dates),5))],-1)
 return z.astype('float32')

def fc_daily(a,dates):
 x=np.stack([a[...,0].mean(2),a[...,1].mean(2),a[...,2].sum(2)],-1)
 return pd.DataFrame(x.reshape(-1,3),index=pd.MultiIndex.from_product([TAS,dates],names=['ta_id','date']),columns=['temp','rhu','rain'])

def corr(a,b):
 v=np.isfinite(a)&np.isfinite(b)
 if v.sum()<30:return np.nan
 if np.std(a[v])<1e-8 or np.std(b[v])<1e-8:return 0.
 return float(np.corrcoef(a[v],b[v])[0,1])

def map_observed(obs,fc,dates,k):
 hist=dates[max(0,k-180):k];out=[];mapping={}
 for ta in TAS:
  f=fc.loc[ta].reindex(hist);scores=[]
  for st in sorted(obs.index.get_level_values(0).unique()):
   o=obs.loc[st].reindex(hist)
   value=.5*corr(f.temp.diff().to_numpy(),o.temp.diff().to_numpy())+.3*corr((f.rhu-f.rhu.rolling(15,center=True,min_periods=5).mean()).to_numpy(),(o.rhu-o.rhu.rolling(15,center=True,min_periods=5).mean()).to_numpy())+.2*corr(np.log1p(f.rain.clip(lower=0).to_numpy()),np.log1p(o.rain.clip(lower=0).to_numpy()))
   if np.isfinite(value):scores.append((value,st))
  if not scores:raise ValueError('Insufficient actual/forecast overlap for '+ta)
  score,st=sorted(scores,key=lambda t:(-t[0],t[1]))[0];mapping[ta]={'station':st,'similarity':score,'fit_end':str(dates[k-1].date())}
  out.append(obs.loc[st].reindex(dates)[['temp','rhu','rain']].to_numpy(float))
 y=np.stack(out);y[...,2]=np.log1p(np.maximum(y[...,2],0));return y,mapping

def fit_bridge(a,obs,dates,k):
 envelope=clean_fit(a[:,:k]);x=raw_features(a,envelope,dates);y,mapping=map_observed(obs,fc_daily(a,dates),dates,k)
 # Regional identity; all normalizers and all fits stop before the forecast origin.
 x=np.concatenate([x,np.broadcast_to(np.eye(9)[:,None,:],(9,len(dates),9))],-1)
 models=[];res=[]
 for j in range(3):
  xx=x[:,:k].reshape(-1,x.shape[-1]);yy=y[:,:k,j].reshape(-1);v=np.isfinite(yy)
  m=ExtraTreesRegressor(n_estimators=120,min_samples_leaf=6,max_features=.8,n_jobs=4,random_state=42+j).fit(xx[v],yy[v]);models.append(m)
  # Training residual provides only a starting scale. Neural NLL calibrates it on OOF errors.
  res.append(max(float(np.sqrt(np.mean((m.predict(xx[v])-yy[v])**2))),[1.,3.,.25][j]))
 return dict(models=models,envelope=envelope,residual=np.array(res),mapping=mapping,fit_end=str(dates[k-1].date())),y

def bridge_predict(model,a,dates):
 x=raw_features(a,model['envelope'],dates);x=np.concatenate([x,np.broadcast_to(np.eye(9)[:,None,:],(9,len(dates),9))],-1);xx=x.reshape(-1,x.shape[-1])
 mu=[];sd=[]
 for j,m in enumerate(model['models']):
  tree=np.stack([t.predict(xx) for t in m.estimators_]);mu.append(tree.mean(0));sd.append(np.sqrt(tree.var(0)+model['residual'][j]**2))
 return np.stack(mu,-1).reshape(9,len(dates),3),np.stack(sd,-1).reshape(9,len(dates),3)

def make_oof(a,obs,dates):
 mu=np.zeros((9,len(dates),3));sd=np.zeros_like(mu);target=np.full_like(mu,np.nan);available=np.zeros((9,len(dates),1));audit=[];bridges={}
 # First two months are history only. First trainable horizon begins December 1.
 mu[...,0]=a[...,0].mean(2);mu[...,1]=a[...,1].mean(2);mu[...,2]=np.log1p(np.maximum(a[...,2].sum(2),0));sd[:]=[5,15,1]
 for date in pd.date_range('2024-12-01',dates[-1],freq='MS'):
  k=dates.get_loc(date);end=min(k+date.days_in_month,len(dates));model,y=fit_bridge(a,obs,dates,k)
  mm,ss=bridge_predict(model,a[:,k:end],dates[k:end]);mu[:,k:end]=mm;sd[:,k:end]=ss;target[:,k:end]=y[:,k:end];available[:,k:end]=1;bridges[str(date.date())]=model
  audit.append({'origin':str(date.date()),'fit_end':model['fit_end'],'pred_end':str(dates[end-1].date()),'mapping':model['mapping']});print('OOF weather',date.date(),flush=True)
 return dict(mu=mu,sd=sd,target=target,available=available,audit=audit,bridges=bridges)

def load_training(root,actual_path,actual_reader=None):
 dates=pd.date_range(START,'2025-09-30');load=pd.read_csv(Path(root)/'train_data/load_data/pv_load_train.csv',parse_dates=['date']);load=load[load.date.isin(dates)]
 pvs=sorted(load.pv_id.unique());expected=sorted('pv_'+str(i) for i in range(1,101))
 if pvs!=expected or load.duplicated(['pv_id','date']).any() or not load.groupby('pv_id').ta_id.nunique().eq(1).all():raise ValueError('Invalid PV identity/mapping')
 if len(load)!=100*len(dates):raise ValueError('Missing PV/day rows')
 y=load.pivot(index='pv_id',columns='date',values='load').reindex(index=pvs,columns=dates).to_numpy(float)
 if np.isinf(y).any() or np.any(y[np.isfinite(y)]<0):raise ValueError('Invalid historical PV load')
 ta=np.array([TAS.index(t) for t in load.groupby('pv_id').first().reindex(pvs).ta_id]);a=read_hours(Path(root)/'train_data/weather_data/pv_weather_train_hourly.csv',dates)
 return dict(y=y,a=a,dates=dates,pvs=np.array(pvs),ta=ta),(actual_reader or observed_daily)(actual_path)

"""V18-compatible 17 daily/diurnal observed targets, forecast-issued multi-output bridge."""
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
FIELDS=['temp','temp_max','temp_min','temp_range','rhu','rhu_min','rhu_max','rain','temp_day','rhu_day','rain_day','temp_noon','rhu_noon','rain_noon','temp_night','rhu_night','rain_night']
RAIN=[7,10,13,16]

def aggregate17(h,region='station',temp='TEM',rh='RHUM',rain='PRECIPITATION',observed=True):
 h=h.copy();h['date']=h.datetime.dt.normalize();h['hour']=h.datetime.dt.hour;g=h.groupby([region,'date'])
 out=g.agg(temp=(temp,'mean'),temp_max=(temp,'max'),temp_min=(temp,'min'),rhu=(rh,'mean'),rhu_min=(rh,'min'),rhu_max=(rh,'max'),rain=(rain,'sum'))
 out['temp_range']=out.temp_max-out.temp_min
 if observed:
  for prefix,raw in [('temp',temp),('rhu',rh),('rain',rain)]:out.loc[g[raw].count()<20,[c for c in out if c.startswith(prefix)]]=np.nan
 for name,mask in [('day',h.hour.between(6,18)),('noon',h.hour.between(10,15)),('night',h.hour.between(0,5))]:
  q=h[mask].groupby([region,'date']).agg(temp=(temp,'mean'),rhu=(rh,'mean'),rain=(rain,'sum'));out=out.join(q.add_suffix('_'+name))
 return out[FIELDS]

def observed17(path):
 parts=[]
 for c in pd.read_csv(path,usecols=['DATETIME','TQ_ID','TEM','RHUM','PRECIPITATION'],chunksize=250000):
  dt=pd.to_datetime(c.DATETIME);take=dt.between(START,'2025-09-30 23:00:00')&(dt.dt.minute==0)&(dt.dt.second==0);q=c.loc[take].copy();q['datetime']=dt[take];q['station']='ta:'+q.TQ_ID.astype(str)
  for f,lo,hi in [('TEM',-60,65),('RHUM',0,100),('PRECIPITATION',0,9998)]:
   v=pd.to_numeric(q[f]);q[f]=v.where(np.isfinite(v)&v.between(lo,hi))
  parts.append(q[['station','datetime','TEM','RHUM','PRECIPITATION']])
 h=pd.concat(parts,ignore_index=True);dup=h.duplicated(['station','datetime'],keep=False)
 if dup.any() and h[dup].groupby(['station','datetime'])[['TEM','RHUM','PRECIPITATION']].nunique(dropna=False).gt(1).any().any():raise ValueError('Conflicting TA actual observations')
 return aggregate17(h.drop_duplicates(['station','datetime']))

def forecast17(a,dates):
 h=pd.DataFrame(a.reshape(-1,9),columns=WX);h['ta_id']=np.repeat(TAS,len(dates)*24);h['datetime']=np.tile(pd.date_range(dates[0],periods=len(dates)*24,freq='h'),9)
 out=aggregate17(h,'ta_id','TEM','RHU','PRE_15m',False)
 return np.stack([out.loc[t].reindex(dates).to_numpy(float) for t in TAS])

def transform17(y):
 z=np.array(y,float,copy=True);z[...,RAIN]=np.log1p(np.maximum(z[...,RAIN],0));return z

def fit_bridge17(a,obs,dates,k):
 env=clean_fit(a[:,:k]);raw=raw_features(a,env,dates);x=np.concatenate([raw,np.broadcast_to(np.eye(9)[:,None,:],(9,len(dates),9))],-1)
 _,mapping=map_observed(obs,fc_daily(a,dates),dates,k);target=transform17(np.stack([obs.loc[mapping[t]['station']].reindex(dates)[FIELDS].to_numpy(float) for t in TAS]));yy=target[:,:k].reshape(-1,17);xm=x[:,:k].reshape(-1,51)
 ym=_v48_nanmean(yy,0);ys=np.maximum(_v48_nanstd(yy,0),.1);v=np.isfinite(yy).all(1)
 if v.sum()<100:raise ValueError('Insufficient complete 17-field observed weather rows')
 model=ExtraTreesRegressor(n_estimators=160,min_samples_leaf=3,max_features=.8,n_jobs=4,random_state=42).fit(xm[v],(yy[v]-ym)/ys)
 residual=np.maximum(np.sqrt(np.mean((model.predict(xm[v])*ys+ym-yy[v])**2,0)),ys*.1)
 return dict(model=model,envelope=env,mean=ym,std=ys,residual=residual,mapping=mapping,fit_end=str(dates[k-1].date())),target

def bridge_predict17(model,a,dates):
 raw=raw_features(a,model['envelope'],dates);x=np.concatenate([raw,np.broadcast_to(np.eye(9)[:,None,:],(9,len(dates),9))],-1).reshape(-1,51)
 values=np.stack([t.predict(x) for t in model['model'].estimators_]);mu=values.mean(0)*model['std']+model['mean'];sd=np.sqrt(values.var(0)*model['std']**2+model['residual']**2)*model.get('uncertainty_multiplier',np.ones(17))
 # Clamp observed-space means and preserve elementary aggregate consistency.
 for j,f in enumerate(FIELDS):
  if f.startswith('rhu'):mu[:,j]=np.clip(mu[:,j],0,100)
  elif f.startswith('rain'):mu[:,j]=np.maximum(mu[:,j],0)
 mu[:,1]=np.maximum(mu[:,1],mu[:,0]);mu[:,2]=np.minimum(mu[:,2],mu[:,0]);mu[:,3]=mu[:,1]-mu[:,2];mu[:,5]=np.minimum(mu[:,5],mu[:,4]);mu[:,6]=np.maximum(mu[:,6],mu[:,4])
 for j in [10,13,16]:mu[:,j]=np.minimum(mu[:,j],mu[:,7])
 return mu.reshape(9,len(dates),17),sd.reshape(9,len(dates),17)

def make_oof17(a,obs,dates):
 targets=np.full((9,len(dates),17),np.nan);means=np.full_like(targets,np.nan);base_sd=np.full_like(targets,np.nan);bridges={};audit=[]
 for date in pd.date_range('2024-12-01',dates[-1],freq='MS'):
  k=dates.get_loc(date);end=min(k+date.days_in_month,len(dates));model,y=fit_bridge17(a,obs,dates,k);model['uncertainty_multiplier']=uncertainty17(targets[:,:k],means[:,:k],base_sd[:,:k]);mu,sd=bridge_predict17(model,a[:,k:end],dates[k:end]);means[:,k:end]=mu;base_sd[:,k:end]=sd/model['uncertainty_multiplier'];bridges[str(date.date())]=model;targets[:,k:end]=y[:,k:end];audit.append(dict(origin=str(date.date()),fit_end=model['fit_end'],pred_end=str(dates[end-1].date()),mapping=model['mapping']));print('17-field OOF',date.date(),flush=True)
 return dict(target=targets,mean=means,base_sd=base_sd,bridges=bridges,audit=audit)

def state17(d,oof,k):
 scale=_v48_nanmedian(np.where(d['y'][:,:k]>0,d['y'][:,:k],np.nan),1);scale=np.where(np.isfinite(scale)&(scale>0),scale,1.)
 env=clean_fit(d['a'][:,:k]);raw=raw_features(d['a'][:,:k],env,d['dates'][:k]);rm=raw.mean((0,1));rs=raw.std((0,1));rs=np.where(rs>.01,rs,1.)
 ym=_v48_nanmean(oof['target'][:,:k],(0,1));ys=np.maximum(_v48_nanstd(oof['target'][:,:k],(0,1)),.1)
 return dict(pvs=d['pvs'],ta=d['ta'],scale=scale,raw_mean=rm,raw_std=rs,weather_mean=ym,weather_std=ys,envelope=env,origin=np.asarray(str((d['dates'][k-1]+pd.Timedelta(days=1)).date())))


def uncertainty17(target,mean,sd):
 valid=np.isfinite(target)&np.isfinite(mean)&np.isfinite(sd)
 count=valid.sum((0,1));num=np.where(valid,(target-mean)**2,0).sum((0,1));den=np.where(valid,sd**2,0).sum((0,1))
 return np.where(count>=30,np.clip(np.sqrt(num/np.maximum(den,1e-8)),.5,4),1.)

"""Independent pooled weather-to-power experts with issue-time level features."""
import numpy as np
from joblib import Parallel,delayed
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor,HistGradientBoostingRegressor

def context(y,scale,k):
 v=y[:,max(0,k-62):k]/scale[:,None];v=np.where(np.isfinite(v),v,np.nan)
 parts=[]
 for n in [3,7,14,28,62]:
  z=v[:,-n:];parts.extend([_v48_nanmean(z,1),_v48_nanstd(z,1)])
 parts.extend([_v48_nanquantile(v,.1,axis=1),_v48_nanquantile(v,.9,axis=1)])
 return np.nan_to_num(np.stack(parts,-1),nan=1,posinf=1).astype('float32')

def expert_inputs(state,a,dates,mu,sd,ctx,horizon_offset=0):
 raw=raw_features(a,state['envelope'],dates);f=np.concatenate([np.clip((raw-state['raw_mean'])/state['raw_std'],-8,8),(mu-state['weather_mean'])/state['weather_std'],np.clip(sd/state['weather_std'],.025,5)],-1)[state['ta']]
 n=len(dates);c=np.broadcast_to(ctx[:,None,:],(100,n,12));ident=np.broadcast_to(np.stack([np.arange(100),state['ta']],-1)[:,None,:],(100,n,2));lead=np.broadcast_to(np.arange(horizon_offset,horizon_offset+n)[None,:,None]/31,(100,n,1))
 return np.concatenate([f,c,ident,lead],-1).reshape(100*n,-1)

def expert_train_data(d,oof,k,state):
 xs=[];ys=[];weights=[];dates=d['dates'];y=d['y'];scale=state['scale'];cache={}
 for s in range(7,k-7,14):
  n=min(31,k-s);keys=[key for key in oof['bridges'] if pd.Timestamp(key)<=dates[s]]
  if keys:mu,sd=bridge_predict17(oof['bridges'][max(keys)],d['a'][:,s:s+n],dates[s:s+n])
  else:
   mu=transform17(forecast17(d['a'][:,s:s+n],dates[s:s+n]));sd=np.broadcast_to(np.array([5,5,5,3,15,15,15,1,5,15,1,5,15,1,5,15,1]),mu.shape)
  ctx=context(y,scale,s)
  x=expert_inputs(state,d['a'][:,s:s+n],dates[s:s+n],mu,sd,ctx);yy=y[:,s:s+n]/scale[:,None];v=np.isfinite(yy)&(yy>=0)
  xs.append(x[v.ravel()]);ys.append(yy[v]);weights.append(np.full(v.sum(),1.))
 return np.concatenate(xs),np.concatenate(ys),np.concatenate(weights)

def fit_expert(x,y,w,kind):
 if kind=='local_tree':
  def fit_user(u):
   v=x[:,88]==u
   return ExtraTreesRegressor(n_estimators=300,min_samples_leaf=3,max_features=.8,n_jobs=1,random_state=42).fit(x[v,:88],np.log(np.maximum(y[v],.001)),sample_weight=w[v])
  return Parallel(n_jobs=6,prefer='threads')(delayed(fit_user)(u) for u in range(100))
 if kind=='pooled_tree':m=ExtraTreesRegressor(n_estimators=300,min_samples_leaf=6,max_features=.8,n_jobs=6,random_state=42)
 elif kind=='pooled_boost':m=HistGradientBoostingRegressor(max_iter=220,max_leaf_nodes=23,min_samples_leaf=30,l2_regularization=10,learning_rate=.045,early_stopping=False,categorical_features=[88,89],random_state=42)
 else:raise ValueError(kind)
 m.fit(x,np.log(np.maximum(y,.001)),sample_weight=w);return m

def expert_predict(m,x,state,n):
 if isinstance(m,list):return np.stack([np.exp(mm.predict(x[x[:,88]==u,:88])) for u,mm in enumerate(m)])*state['scale'][:,None]
 return np.exp(m.predict(x)).reshape(100,n)*state['scale'][:,None]
def observed_for(bridge, dates):
    return transform17(np.stack([OBS.loc[bridge['mapping'][t]['station']].reindex(dates)[FIELDS].to_numpy(float) for t in TAS]))

def train_arrays(k, state):
    xs = {n: [] for n in ['corrected', 'raw', 'actual']}
    ys = []
    dates = D['dates']
    scale = state['scale']
    missing = 0
    for s in range(7, k - 7, 14):
        n = min(31, k - s)
        dd = dates[s:s + n]
        a = D['a'][:, s:s + n]
        keys = [key for key in O['bridges'] if pd.Timestamp(key) <= dates[s]]
        raw = transform17(forecast17(a, dd))
        if keys:
            bridge = O['bridges'][max(keys)]
            mu, sd = bridge_predict17(bridge, a, dd)
            actual = observed_for(bridge, dd)
        else:
            mu = raw
            sd = np.broadcast_to(np.array([5, 5, 5, 3, 15, 15, 15, 1, 5, 15, 1, 5, 15, 1, 5, 15, 1]), mu.shape)
            actual = observed_for(O['bridges'][str(dates[k].date())], dd)
        missing += int((~np.isfinite(actual)).sum())
        actual = np.where(np.isfinite(actual), actual, mu)
        ctx = context(D['y'], scale, s)
        yy = D['y'][:, s:s + n] / scale[:, None]
        valid = np.isfinite(yy) & (yy >= 0)
        for name, w in [('corrected', mu), ('raw', raw), ('actual', actual)]:
            xs[name].append(expert_inputs(state, a, dd, w, sd, ctx)[valid.ravel()])
        ys.append(yy[valid])
    return ({n: np.concatenate(x) for n, x in xs.items()}, np.concatenate(ys), missing)
def trajectory(x, n):
    a = x.reshape(100, n, -1)
    wx = np.concatenate([a[:, :, :9], a[:, :, 42:59]], -1)
    prev = np.concatenate([wx[:, :1], wx[:, :-1]], 1)
    nxt = np.concatenate([wx[:, 1:], wx[:, -1:]], 1)
    return np.concatenate([a, wx - prev, nxt - wx, (prev + wx + nxt) / 3], -1).reshape(100 * n, -1)

def student_data(k, state):
    xs = []
    ys = []
    ts = []
    states = []
    counts = [0, 0]
    for s in range(7, k - 7, 14):
        n = min(31, k - s)
        dd = D['dates'][s:s + n]
        keys = [key for key in O['bridges'] if pd.Timestamp(key) <= dd[0]]
        if keys:
            mu, sd = bridge_predict17(O['bridges'][max(keys)], D['a'][:, s:s + n], dd)
        else:
            mu = transform17(forecast17(D['a'][:, s:s + n], dd))
            sd = np.broadcast_to(np.array([5, 5, 5, 3, 15, 15, 15, 1, 5, 15, 1, 5, 15, 1, 5, 15, 1]), mu.shape)
        x = trajectory(expert_inputs(state, D['a'][:, s:s + n], dd, mu, sd, context(D['y'], state['scale'], s)), n)
        y = D['y'][:, s:s + n] / state['scale'][:, None]
        t = y.copy()
        use = np.zeros(n, bool)
        for j, day in enumerate(dd):
            key = str(day.replace(day=1).date())
            if key in teacher and pd.Timestamp(key) <= dd[0]:
                t[:, j] = teacher[key][:, day.day - 1] / state['scale']
                use[j] = True
        lowcut = _v48_nanquantile(np.where(D['y'][:, :s] > 0, D['y'][:, :s], np.nan), 0.2, axis=1) / state['scale']
        lowcut = np.where(np.isfinite(lowcut), lowcut, 0.3)
        actual_low = (y <= lowcut[:, None]).astype(float)
        soft = 1 / (1 + np.exp(np.clip((np.log(np.maximum(t, 0.001)) - np.log(np.maximum(lowcut[:, None], 0.001))) / 0.25, -30, 30)))
        soft = np.where(use[None, :], 0.5 * actual_low + 0.5 * soft, actual_low)
        prior = np.concatenate([(D['y'][:, s - 1] / state['scale'] <= lowcut)[:, None], actual_low[:, :-1]], 1)
        rec = prior * (y > lowcut[:, None] * 1.5)
        valid = np.isfinite(y) & (y >= 0)
        xs.append(x[valid.ravel()])
        ys.append(np.log(np.maximum(y[valid], 0.001)))
        ts.append(np.log(np.maximum(t[valid], 0.001)))
        states.append(np.stack([soft, rec], -1)[valid])
        counts[0] += int(valid.sum())
        counts[1] += int((valid & use[None, :]).sum())
    return (np.concatenate(xs), np.concatenate(ys), np.concatenate(ts), np.concatenate(states), counts)
def make_teachers(d,oof,obs):
 from concurrent.futures import ThreadPoolExecutor,as_completed
 def build(date):
  k=d['dates'].get_loc(date);n=pd.Timestamp(date).days_in_month;dd=d['dates'][k:k+n];bridge=oof['bridges'][date];state=state17(d,oof,k)
  for key,fallback in [('weather_mean',bridge['mean']),('weather_std',bridge['std'])]:state[key]=np.where(np.isfinite(state[key]),state[key],fallback)
  xs,y,_=train_arrays(k,state);mu,sd=bridge_predict17(bridge,d['a'][:,k:k+n],dd);act=observed_for(bridge,dd);f=expert_inputs(state,d['a'][:,k:k+n],dd,np.where(np.isfinite(act),act,mu),sd,context(d['y'],state['scale'],k));pr=[]
  for kind in ['pooled_tree','local_tree']:
   m=fit_expert(xs['actual'],y,np.ones(len(y)),kind);pr.append(expert_predict(m,f,state,n));del m
  print('[PV V48 teacher]',date,flush=True);return date,np.mean(pr,0)
 dates=[str(d.date()) for d in pd.date_range('2024-12-01','2025-09-01',freq='MS')];teachers={}
 with ThreadPoolExecutor(max_workers=3) as pool:
  for future in as_completed([pool.submit(build,date) for date in dates]):
   date,pred=future.result();teachers[date]=pred
 return teachers

def fit_student(x,target):
 pooled=ExtraTreesRegressor(n_estimators=300,min_samples_leaf=6,max_features=.8,n_jobs=6,random_state=42).fit(x,target)
 def fit_user(u):
  v=x[:,88]==u
  return ExtraTreesRegressor(n_estimators=300,min_samples_leaf=3,max_features=.8,n_jobs=1,random_state=42).fit(x[v],target[v])
 local=Parallel(n_jobs=6,prefer='threads')(delayed(fit_user)(u) for u in range(100))
 return {'pooled_tree':pooled,'local_tree':local}

def student_predict(experts,x,state,n):
 pooled=np.exp(experts['pooled_tree'].predict(x)).reshape(100,n)
 local=np.stack([np.exp(m.predict(x[x[:,88]==u])) for u,m in enumerate(experts['local_tree'])])
 return .5*(pooled+local)*state['scale'][:,None]


def _v48_nan_reduce(a, axis, operation, q=None):
    a = np.asarray(a)
    available = np.any(~np.isnan(a), axis=axis, keepdims=True)
    present = np.squeeze(available, axis=axis)
    if a.size == 0:
        return np.full(present.shape, np.nan)
    # Only empty slices get temporary values. Their result remains NaN so
    # existing bridge/context/scale fallbacks retain exactly their semantics.
    safe = np.where(available, a, 0)
    reducer = getattr(np, 'nan' + operation)
    result = reducer(safe, axis=axis) if q is None else reducer(safe, q, axis=axis)
    return np.where(present, result, np.nan)

def _v48_nanmean(a, axis=None):
    return _v48_nan_reduce(a, axis, 'mean')

def _v48_nanstd(a, axis=None):
    return _v48_nan_reduce(a, axis, 'std')

def _v48_nanmedian(a, axis=None):
    return _v48_nan_reduce(a, axis, 'median')

def _v48_nanquantile(a, q, axis=None):
    return _v48_nan_reduce(a, axis, 'quantile', q)
