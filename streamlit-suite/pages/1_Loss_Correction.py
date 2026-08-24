import io, hashlib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.optimize import differential_evolution

st.set_page_config(page_title='Loss Correction - Solar Suite', page_icon='☀️', layout='wide')

GHI_COLS = ['GHI C11','GHI C12','GHI C13','GHI C14','GHI C15']
CLUSTERS = ['C11','C12','C13','C14','C15']
BOUNDS = [(0,10),(10,30),(65,80),(47,53),(10,70),(10,70)]


def norm(x):
    return ''.join(str(x).replace('\n',' ').replace('\xa0',' ').split()).lower()


def find_col(df, *names):
    m = {norm(c): c for c in df.columns}
    for name in names:
        if norm(name) in m: return m[norm(name)]
    return None


def num(x):
    if isinstance(x, pd.Series):
        return pd.to_numeric(x, errors='coerce').fillna(0).to_numpy(float)
    return pd.to_numeric(pd.Series(x), errors='coerce').fillna(0).to_numpy(float)


def arr(x):
    return np.asarray(x, dtype=np.float64)


def clean_columns(df):
    df = df.copy()
    df.columns = [str(c).strip().replace('\xa0',' ') for c in df.columns]
    return df


def get_area(raw):
    d = clean_columns(pd.read_excel(io.BytesIO(raw), sheet_name='Area & Efficiency', header=1))
    c_cluster = find_col(d, 'Clusters *', 'Clusters', 'Clusers')
    c_eff = find_col(d, 'Standard PV Efficiency (%) *', 'Standard PV Efficiency (%)')
    c_n = find_col(d, 'No of Module *', 'No of Module')
    c_a = find_col(d, 'Area of 1 Module (m2) *', 'Area of 1 Module (m2)')
    c_total = find_col(d, 'Total area (m2) *', 'Total area (m2)', 'Total area(m2)')
    if not all([c_cluster,c_eff,c_n,c_a]):
        raise ValueError('Required Area & Efficiency columns not found. Expected cluster, Standard PV Efficiency, No of Module and Area of 1 Module.')
    d = d[d[c_cluster].astype(str).str.strip().isin(CLUSTERS)].copy()
    d[c_eff], d[c_n], d[c_a] = num(d[c_eff]), num(d[c_n]), num(d[c_a])
    d['Total area (m2)'] = num(d[c_total]) if c_total else d[c_n]*d[c_a]
    d['Cluster'] = d[c_cluster].astype(str).str.strip()
    d = d[(d['Total area (m2)'] > 0) & (d[c_eff] > 0)].reset_index(drop=True)
    if d.empty: raise ValueError('No valid module rows found in Area & Efficiency.')
    return d, c_eff


def get_lat(raw):
    d = clean_columns(pd.read_excel(io.BytesIO(raw), sheet_name='Forecast Config', header=8))
    c = find_col(d, 'Lat')
    if c is None:
        # fallback: scan all cells for first plausible latitude
        for col in d.columns:
            v = pd.to_numeric(d[col], errors='coerce').dropna()
            if len(v) and v.iloc[0] != 0 and abs(v.iloc[0]) <= 90:
                return float(v.iloc[0])
        raise ValueError('Latitude (Lat) not found in Forecast Config.')
    v = pd.to_numeric(d[c], errors='coerce').dropna()
    if v.empty: raise ValueError('Latitude value is invalid.')
    return float(v.iloc[0])


def get_tilt(raw):
    # This workbook has the Month label on header row 6 and Fixed values below it.
    d = clean_columns(pd.read_excel(io.BytesIO(raw), sheet_name='Config Tilt Angle', header=6))
    month = find_col(d, 'Month')
    if month is None: raise ValueError('Month column missing in Config Tilt Angle.')
    candidates = []
    for c in d.columns:
        if c == month: continue
        vals = pd.to_numeric(d[c], errors='coerce')
        marker = str(d[c].iloc[0]).strip().lower() if len(d) else ''
        if marker == 'fixed' and vals.notna().sum() >= 3: candidates.append(c)
    if not candidates:
        for c in d.columns:
            if c == month: continue
            vals = pd.to_numeric(d[c], errors='coerce')
            if vals.notna().sum() >= 3: candidates.append(c)
    if not candidates: raise ValueError('Fixed tilt values not found in Config Tilt Angle.')
    fixed = candidates[0]
    out = {}
    for _, r in d.iterrows():
        m = str(r[month]).strip()
        v = pd.to_numeric(pd.Series([r[fixed]]), errors='coerce').iloc[0]
        if m.lower() in {x.lower() for x in ['January','February','March','April','May','June','July','August','September','October','November','December']} and pd.notna(v):
            out[m] = float(v)
    if not out: raise ValueError('No monthly Fixed tilt values found.')
    return out


def get_input(raw):
    result = clean_columns(pd.read_excel(io.BytesIO(raw), sheet_name='Result', header=0))
    missing = [c for c in GHI_COLS if find_col(result,c) is None]
    if missing: raise ValueError(f'Missing GHI columns in Result: {missing}')
    fixed = clean_columns(pd.read_excel(io.BytesIO(raw), sheet_name='Fixed-C11', header=1))
    actual_col = find_col(fixed, 'Actual', 'Actual Power', 'Actual Power MW')
    ghi = np.column_stack([num(result[find_col(result,c)]) for c in GHI_COLS])
    actual = num(fixed[actual_col]) if actual_col else None
    date_col = find_col(fixed, 'Date')
    dates = pd.to_datetime(fixed[date_col], errors='coerce').dropna() if date_col else pd.Series(dtype='datetime64[ns]')
    calc_date = dates.iloc[0] if len(dates) else pd.Timestamp.today()
    if actual is None: raise ValueError(f"Actual column not found in Fixed-C11. Available columns: {list(fixed.columns)}")
    n = min(96, len(actual), len(ghi))
    if n <= 0: raise ValueError('No GHI/Actual rows available.')
    return ghi[:n], actual[:n], pd.Timestamp(calc_date)


def geometry(lat, tilt, calc_date):
    day = pd.Timestamp(calc_date).dayofyear
    dec = 23.45*np.sin(np.radians(360*(284+day)/365))
    elev = 90-lat+dec
    sa = np.sin(np.radians(elev)); sab = np.sin(np.radians(elev+tilt))
    return sab/max(abs(sa),1e-9)


def cluster_areas(area, loss):
    std_col = 'std' if 'std' in area.columns else next(c for c in area.columns if norm(c).startswith('standardpvefficiency'))
    std = num(area[std_col])
    eff = area['Total area (m2)'].to_numpy(float) * np.maximum(std - float(loss), 0) / 100
    return np.array([eff[area['Cluster'].astype(str).str.strip().eq(c)].sum() for c in CLUSTERS], float)


def optimize_loss(area, ghi, actual, factor, cluster=True):
    std_col = next(c for c in area.columns if norm(c).startswith('standardpvefficiency'))
    std = num(area[std_col]); a = area['Total area (m2)'].to_numpy()
    losses = np.round(np.arange(0, max(0.1, float(std.min())-0.1)+0.001, 0.1),1)
    if actual.max() <= 0: return 0.0
    peaks=[]
    for loss in losses:
        ca = cluster_areas(area, loss)
        pred = (ghi*factor*ca[None,:]).sum(axis=1)/1e6
        peaks.append(abs(actual.max()-pred.max()))
    return float(losses[int(np.argmin(peaks))])


def fixed_forecast(ghi, area, loss, factor):
    ca = cluster_areas(area, loss)
    return (ghi*factor*ca[None,:]).sum(axis=1)/1e6


def tracking_forecast(ghi, area, loss, params):
    dhi,start,end,maximum,east,west = map(int, params)
    if not start < maximum < end: raise ValueError('Tracking condition: Starting Block < Max Block < Ending Block.')
    blocks=np.arange(1,ghi.shape[0]+1,dtype=float)
    m1=90/(start-1-maximum); m2=90/(end+1-maximum)
    zen=np.where(blocks<=maximum,np.minimum(89,m1*(blocks-maximum)),np.minimum(89,m2*(blocks-maximum)))
    panel=np.where(blocks<maximum,np.minimum(zen,abs(east)),np.where((blocks>maximum)&(zen>west),west,zen))
    cos=np.clip(np.cos(np.radians(panel)),1e-6,None)
    ca=cluster_areas(area,loss)
    return (ghi*(1-dhi/100)*ca[None,:]/cos[:,None]).sum(axis=1)/1e6


def tracking_objective(actual, ghi, area, loss):
    ca=cluster_areas(area,loss); mask=actual!=0
    a=actual[mask]
    if a.size==0 or a.max()<=0: raise ValueError('No non-zero Actual values for Tracking.')
    peak=a.max(); energy=a.sum(); blocks=np.arange(1,ghi.shape[0]+1,dtype=float)
    def f(x):
        DHI,start,end,maximum,east,west=np.rint(x).astype(int)
        if not start<maximum<end:return 1e9
        m1=90/(start-1-maximum);m2=90/(end+1-maximum)
        zen=np.where(blocks<=maximum,np.minimum(89,m1*(blocks-maximum)),np.minimum(89,m2*(blocks-maximum)))
        panel=np.where(blocks<maximum,np.minimum(zen,abs(east)),np.where((blocks>maximum)&(zen>west),west,zen))
        cos=np.clip(np.cos(np.radians(panel)),1e-6,None)
        pred=(ghi*(1-DHI/100)*ca[None,:]/cos[:,None]).sum(axis=1)/1e6
        p=pred[mask]
        if not np.isfinite(p).all(): return 1e9
        return .8*np.mean(np.abs(a-p))/peak + .1*abs(peak-p.max())/peak + .1*abs(energy-p.sum())/energy
    return f


@st.cache_data(show_spinner=False, max_entries=3)
def optimize_tracking_cached(actual_t, ghi_t, area_t, loss):
    actual=np.asarray(actual_t,float); ghi=np.asarray(ghi_t,float)
    # Compact tuples keep Streamlit cache hashing stable.
    area=pd.DataFrame({'Cluster':np.asarray(area_t[0]),'Total area (m2)':np.asarray(area_t[1]),'std':np.asarray(area_t[2])})
    obj=tracking_objective(actual,ghi,area,loss)
    r=differential_evolution(obj,BOUNDS,maxiter=20,popsize=6,tol=.005,polish=False,seed=42,workers=1,updating='immediate')
    return tuple(np.rint(r.x).astype(int).tolist())


def metrics(a,p):
    m=a>0
    if not m.any(): return {'mae':0,'rmse':0,'peak':0,'energy':0}
    x,y=a[m],p[m]
    return {'mae':float(np.mean(abs(x-y))),'rmse':float(np.sqrt(np.mean((x-y)**2))), 'peak':float(abs(x.max()-y.max())/x.max()*100),'energy':float(abs(x.sum()-y.sum())/x.sum()*100)}


def chart(a,p,title):
    n=min(len(a),len(p)); f=go.Figure(); f.add_scatter(x=np.arange(1,n+1),y=p[:n],mode='lines',name='Forecast'); f.add_scatter(x=np.arange(1,n+1),y=a[:n],mode='lines',name='Actual'); f.update_layout(title=title,height=430,template='plotly_white',hovermode='x unified',xaxis_title='15 Minute Block',yaxis_title='Power (MW)'); return f


def run_fixed(area,ghi,actual,tilt,lat,calc_date,loss=None):
    factor=geometry(lat,tilt,calc_date)
    best=optimize_loss(area,ghi,actual,factor) if loss is None else float(loss)
    pred=fixed_forecast(ghi,area,best,factor)
    return best,pred


def run_tracking(area,ghi,actual,tilt,lat,loss,params):
    # Tracking keeps the original tracking geometry, so tilt is not used in this stage.
    pred=tracking_forecast(ghi,area,loss,params)
    return pred


# ---------------- UI ----------------
st.title('☀️ Solar Forecast Loss Correction')
st.caption('Workbook is read once. Heavy optimization runs only after an explicit button.')
uploaded=st.file_uploader('Solar Excel File',type=['xlsx','xls'],label_visibility='collapsed')
if uploaded is None:
    st.info('Upload the Solar Excel file to start.'); st.stop()
raw=uploaded.getvalue(); sig=hashlib.sha256(raw).hexdigest()
if st.session_state.get('lc_sig')!=sig:
    for k in ['lc_wb','lc_run','lc_tracking','lc_best_loss','lc_params']:
        st.session_state.pop(k,None)
    st.session_state.lc_sig=sig

if 'lc_wb' not in st.session_state:
    try:
        with st.spinner('Reading workbook once...'):
            area,effcol=get_area(raw); lat=get_lat(raw); tilt_map=get_tilt(raw); ghi,actual,calc_date=get_input(raw)
        st.session_state.lc_wb={'area':area,'lat':lat,'tilt':tilt_map,'ghi':ghi,'actual':actual,'date':calc_date}
    except Exception as e:
        st.error(f'Input preparation failed: {e}'); st.stop()

wb=st.session_state.lc_wb
st.subheader('GHI / Actual Input')
inp=pd.DataFrame(ghi=[]) if False else pd.DataFrame(wb['ghi'],columns=GHI_COLS)
inp['Actual']=wb['actual']
edited=st.data_editor(inp,height=260,num_rows='fixed',hide_index=True,width='stretch',key='lc_editor')
plant=st.segmented_control('Plant Type',['Fixed','Tracking'],default='Fixed',width='stretch') or 'Fixed'

if st.button('⚡ Run Automatic Calculation',type='primary',width='stretch'):
    st.session_state.lc_run=True
    st.session_state.lc_tracking=None
    st.session_state.lc_params=None
    wb['ghi']=edited[GHI_COLS].apply(pd.to_numeric,errors='coerce').fillna(0).to_numpy(float)
    wb['actual']=pd.to_numeric(edited['Actual'],errors='coerce').fillna(0).to_numpy(float)
    month=pd.Timestamp.today().strftime('%B'); tilt=float(wb['tilt'].get(month,0))
    try:
        best,pred=run_fixed(wb['area'],wb['ghi'],wb['actual'],tilt,wb['lat'],wb['date'])
        st.session_state.lc_best_loss=best
        if plant=='Tracking': st.session_state.lc_tracking='ready'
        else: st.session_state.lc_tracking='done'
    except Exception as e:
        st.error(f'Calculation failed: {e}'); st.stop()

if not st.session_state.get('lc_run',False):
    st.info('Edit the input, select Fixed or Tracking, then click Run Automatic Calculation.'); st.stop()

area=wb['area']; ghi=wb['ghi']; actual=wb['actual']; tilt=float(wb['tilt'].get(wb['date'].strftime('%B'),0)); best=st.session_state.get('lc_best_loss',0.0)

if plant=='Fixed':
    loss=st.number_input('Efficiency Loss (%)',0.0,50.0,float(best),0.1,key='fixed_loss')
    pred=fixed_forecast(ghi,area,loss,geometry(wb['lat'],tilt,wb['date'])); m=metrics(actual,pred)
    c=st.columns(4); c[0].metric('Efficiency Loss',f'{loss:.1f}%'); c[1].metric('MAE',f"{m['mae']:.3f}"); c[2].metric('Peak Error',f"{m['peak']:.2f}%"); c[3].metric('Energy Error',f"{m['energy']:.2f}%")
    st.plotly_chart(chart(actual,pred,'Fixed Plant - Forecast vs Actual'),width='stretch')
else:
    if st.session_state.get('lc_params') is None:
        st.info(f'Automatic efficiency loss: {best:.1f}%. Tracking optimization is intentionally manual to prevent freezing.')
        if st.button('🧠 Run Tracking Optimization',type='primary',width='stretch'):
            try:
                std_col=next(c for c in area.columns if norm(c).startswith('standardpvefficiency'))
                area_t=(tuple(area['Cluster']),tuple(area['Total area (m2)']),tuple(num(area[std_col])))
                with st.spinner('Optimizing tracking parameters...'):
                    p=optimize_tracking_cached(tuple(actual),tuple(map(tuple,ghi)),area_t,best)
                st.session_state.lc_params={'DHI':p[0],'start':p[1],'end':p[2],'max':p[3],'east':p[4],'west':p[5],'loss':best}
                st.rerun()
            except Exception as e: st.error(f'Tracking optimization failed: {e}')
        st.stop()
    p=st.session_state.lc_params
    with st.form('tracking_form'):
        loss=st.number_input('Efficiency Loss (%)',0.0,50.0,float(p['loss']),0.1)
        c1,c2,c3=st.columns(3); dhi=c1.number_input('DHI (%)',0,100,p['DHI'],1); start=c2.number_input('Starting Block',1,95,p['start'],1); end=c3.number_input('Ending Block',2,96,p['end'],1)
        c1,c2,c3=st.columns(3); maximum=c1.number_input('Max Block',1,95,p['max'],1); east=c2.number_input('East Limit',0,70,p['east'],1); west=c3.number_input('West Limit',0,70,p['west'],1)
        recalc=st.form_submit_button('🔄 Recalculate',type='primary',width='stretch')
    if recalc:
        if not start<maximum<end: st.error('Condition: Starting Block < Max Block < Ending Block.'); st.stop()
        st.session_state.lc_params={'DHI':dhi,'start':start,'end':end,'max':maximum,'east':east,'west':west,'loss':loss}; p=st.session_state.lc_params
    pred=run_tracking(area,ghi,actual,tilt,wb['lat'],p['loss'],(p['DHI'],p['start'],p['end'],p['max'],p['east'],p['west']))
    m=metrics(actual,pred); c=st.columns(4); c[0].metric('MAE',f"{m['mae']:.3f}"); c[1].metric('RMSE',f"{m['rmse']:.3f}"); c[2].metric('Peak Error',f"{m['peak']:.2f}%"); c[3].metric('Energy Error',f"{m['energy']:.2f}%")
    st.plotly_chart(chart(actual,pred,'Tracking Plant - Forecast vs Actual'),width='stretch')
