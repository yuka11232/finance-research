"""Reproducible first-order triangular fuzzy regression baseline; Python standard library only.
Predict monthly inflation percentage, not CPI index levels. This is a transparent
experimental baseline, not a claim to reproduce a specific published algorithm.
"""
from pathlib import Path
import csv,math,json
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'07_Forecast_Results';OUT.mkdir(exist_ok=True)
def save(name,rows):
 with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def mean(x):return sum(x)/len(x)
def fuzzy_fit(y,k):
 lo,hi=min(y),max(y);pad=max((hi-lo)*.05,.05);lo-=pad;hi+=pad;step=(hi-lo)/(k-1);centers=[lo+i*step for i in range(k)]
 sums=[0.]*k;weights=[0.]*k
 for x,target in zip(y[:-1],y[1:]):
  x=max(lo,min(hi,x))
  for i,c in enumerate(centers):
   w=max(0.,1-abs(x-c)/step);sums[i]+=w*target;weights[i]+=w
 consequents=[sums[i]/weights[i] if weights[i]>0 else mean(y) for i in range(k)]
 return lo,hi,step,centers,consequents
 def_unused=None

def fuzzy_predict(f,x):
 lo,hi,step,centers,consequents=f;x=max(lo,min(hi,x));w=[max(0.,1-abs(x-c)/step) for c in centers]
 return sum(a*b for a,b in zip(w,consequents))/sum(w)
def predict(y,method,k):
 if method=='fuzzy':return fuzzy_predict(fuzzy_fit(y,k),y[-1])
 if method=='last_value':return y[-1]
 if method=='seasonal_naive':return y[-12]
 return mean(y)
def errors(a,p):
 e=[x-y for x,y in zip(a,p)];return mean([abs(v) for v in e]),math.sqrt(mean([v*v for v in e]))
rows=list(csv.DictReader((ROOT/'05_Model_Ready_Data/monthly_broad_CPI_2016-01_to_2026-09.csv').open(encoding='utf-8-sig')))
metrics=[];predictions=[];forecasts=[];spec=[]
for code in sorted(set(r['series'] for r in rows)):
 rr=[r for r in rows if r['series']==code];y=[float(r['mom_inflation_pct']) for r in rr];n=len(y);vstart=n-24;tstart=n-12
 kvals={k:errors(y[vstart:tstart],[predict(y[:i],'fuzzy',k) for i in range(vstart,tstart)])[0] for k in [3,5,7]};k=min(kvals,key=kvals.get)
 valmetrics={}
 for method in ['fuzzy','last_value','seasonal_naive','historical_mean']:
  for split,start,end in [('validation',vstart,tstart),('test',tstart,n)]:
   pp=[predict(y[:i],method,k) for i in range(start,end)];mae,rmse=errors(y[start:end],pp)
   metrics.append(dict(series=code,model=method,split=split,n_forecasts=end-start,MAE_percentage_points=mae,RMSE_percentage_points=rmse,fuzzy_sets=k if method=='fuzzy' else ''))
   if split=='validation':valmetrics[method]=mae
   for i,p in zip(range(start,end),pp):predictions.append(dict(date=rr[i]['date'],series=code,model=method,split=split,actual_mom_pct=y[i],predicted_mom_pct=p))
 selected=min(valmetrics,key=valmetrics.get)
 # Fit once on all available observations; future predictions are never treated as training observations.
 fitted=fuzzy_fit(y,k);history=y[:];series_forecast={}
 for method in ['fuzzy','last_value','seasonal_naive','historical_mean']:
  future=[]
  for h in range(1,7):
   if method=='fuzzy':p=fuzzy_predict(fitted,(future[-1] if future else y[-1]))
   elif method=='last_value':p=y[-1]
   elif method=='seasonal_naive':p=y[n-12+h-1]
   else:p=mean(y)
   future.append(p);month=9+h;year=2026+(month-1)//12;month=(month-1)%12+1
   forecasts.append(dict(date=f'{year}-{month:02d}-01',series=code,model=method,horizon_months=h,forecast_mom_inflation_pct=p,selected_by_validation=int(method==selected),forecast_origin='2026-09-01',status='experimental_point_forecast_not_official'))
 spec.append(dict(series=code,fuzzy_sets=k,selected_by_validation=selected,observations=n,train_observations=vstart,validation_observations=12,test_observations=12))
save('backtest_metrics.csv',metrics);save('backtest_predictions.csv',predictions);save('forecasts_2026-10_to_2027-03.csv',forecasts)
(OUT/'model_specifications.json').write_text(json.dumps(spec,indent=2))
print(json.dumps(spec,indent=2))
