from build_powerbi import *
from build_powerbi import PROJECT_ROOT
import sys,numpy as np
from scipy.stats import norm,beta
repo='data-science-analytics-toolkit';p=PROJECT_ROOT
r=Report(repo,'Customer Support Analytics','Muestra: primeras 50.000 filas del CSV original · una fila por conversación · NLP: primeros 500 mensajes de clientes')
import os
source=Path(os.environ.get('CUSTOMER_SUPPORT_DATA',str(p/'data/customer_support_data.csv')))
with source.open(encoding='utf-8') as f:
 if f.readline().startswith('version https://git-lfs.github.com/spec/'):
  raise ValueError('Descarga git lfs pull o configura CUSTOMER_SUPPORT_DATA con el CSV completo.')
raw=pd.read_csv(source,nrows=50000);d=raw[raw.role=='customer'].drop_duplicates('conv_id').copy();d['Mes']=pd.to_datetime(d.timestamp).dt.to_period('M').astype(str);d['Hora']=pd.to_datetime(d.timestamp).dt.hour;d['Resuelto']=(d.outcome=='Resolved').astype(int);cols=['industry','product','issue_type','language','channel','overall_sentiment','overall_urgency','outcome','primary_intent','Mes','Hora','Resuelto']
r.table('Conversaciones',d[cols],metrics('Conversaciones',[('Conversaciones','COUNTROWS(Conversaciones)'),('Resueltos','SUM(Conversaciones[Resuelto])'),('Resolucion','DIVIDE(SUM(Conversaciones[Resuelto]), COUNTROWS(Conversaciones))'),('Negativos','CALCULATE(COUNTROWS(Conversaciones), Conversaciones[overall_sentiment]="negative")')]))
quality=pd.DataFrame({'Campo':raw.columns,'Nulos':raw.isna().sum().values,'Unicos':raw.nunique().values,'Filas':len(raw)})
r.table('Calidad',quality,metrics('Calidad',[('Nulos','SUM(Calidad[Nulos])'),('Unicos','MAX(Calidad[Unicos])')]))
sys.path.insert(0,str(p/'src/modules/11_nlp_toolkit'));from nlp_toolkit import NLPPipeline
nlp=NLPPipeline(n_temas=4,max_features=500).ejecutar(raw.loc[raw.role=='customer','text'].dropna().head(500).tolist())
for table,key in [('Terminos','terminos_top'),('Bigramas','bigramas'),('Temas','temas')]:
 df=nlp[key].copy();print(table,df.columns.tolist(),flush=True)
 for c in df:
  if df[c].dtype=='object':df[c]=df[c].map(lambda x:', '.join(map(str,x)) if isinstance(x,(list,tuple)) else str(x))
 measures={}
 for c in df.select_dtypes(include='number'):measures['KPI_'+c]=(f'MAX({table}[{c}])','#,0.0000')
 r.table(table,df,measures)
r.table('SentimientosNLP',nlp['sentimientos'].groupby('etiqueta').size().reset_index(name='Mensajes'),metrics('SentimientosNLP',[('Mensajes','SUM(SentimientosNLP[Mensajes])')]))
pairs=[];posterior=[]
for ca in sorted(d.channel.unique()):
 for cb in sorted(d.channel.unique()):
  if ca==cb:continue
  aa=d[d.channel==ca];bb=d[d.channel==cb];na=len(aa);nb=len(bb);sa=aa.Resuelto.sum();sb=bb.Resuelto.sum();pa=sa/na;pb=sb/nb;po=(sa+sb)/(na+nb);se=(po*(1-po)*(1/na+1/nb))**.5;z=(pb-pa)/se if se else 0;pval=2*norm.sf(abs(z));ci_se=(pa*(1-pa)/na+pb*(1-pb)/nb)**.5
  # P(B>A) por integración numérica determinista de las posteriores Beta, sin Monte Carlo.
  from scipy.integrate import quad
  prob=quad(lambda x: beta.pdf(x,sb+1,nb-sb+1)*beta.cdf(x,sa+1,na-sa+1),0,1,points=[pa,pb],epsabs=1e-7)[0]
  pair=ca+' vs '+cb;pairs.append([pair,ca,cb,na,nb,sa,sb,pa,pb,pb-pa,z,pval,pb-pa-1.96*ci_se,pb-pa+1.96*ci_se,prob])
  for channel,s,n in [(ca,sa,na),(cb,sb,nb)]:
   for x in np.linspace(max(0,min(pa,pb)-.1),min(1,max(pa,pb)+.1),120):posterior.append([pair,channel,x,beta.pdf(x,s+1,n-s+1)])
r.table('ComparacionCanales',pd.DataFrame(pairs,columns=['Comparacion','Control','Tratamiento','N_A','N_B','Resueltos_A','Resueltos_B','Tasa_A','Tasa_B','Diferencia','Z','PValue','IC95Inferior','IC95Superior','Prob_B_Supera_A']))
r.table('Posteriores',pd.DataFrame(posterior,columns=['Comparacion','Canal','Tasa','Densidad']),metrics('Posteriores',[('Densidad','MAX(Posteriores[Densidad])')]))
filters=[('Conversaciones','channel'),('Conversaciones','industry'),('Conversaciones','Mes')]
for pg,label,c1,c2 in [('resumen','01 · Resumen ejecutivo','Mes','channel'),('distribucion','02 · Distribución de tickets','language','issue_type'),('sentimientos','03 · Análisis de sentimientos','overall_sentiment','industry'),('urgencia','04 · Urgencia y resultados','overall_urgency','outcome')]:
 r.page(pg,label,filters);r.cards('Conversaciones',list(r.measures['Conversaciones']));r.chart('Conversaciones',c1,'KPI_Conversaciones','Conversaciones · '+c1,30,270,kind='line' if c1=='Mes' else 'bar',legend='overall_sentiment' if pg=='sentimientos' else None);r.chart('Conversaciones',c2,'KPI_Resolucion','Tasa de resolución · '+c2,650,270);r.tablevisual('Conversaciones',[c1,c2,'KPI_Conversaciones','KPI_Resueltos','KPI_Resolucion','KPI_Negativos'],'Detalle de resultados',30,570,1220,260)
r.page('nlp','05 · Análisis de texto NLP',[],note='NLP del motor original sobre 500 mensajes. Solo estadísticas y términos; texto libre e inferencia permanecen en Streamlit.')
c=next(c for c in r.tables['Terminos'] if not pd.api.types.is_numeric_dtype(r.tables['Terminos'][c]));m=next(iter(r.measures['Terminos']));r.chart('Terminos',c,m,'Términos TF-IDF',30,160);r.chart('SentimientosNLP','etiqueta','KPI_Mensajes','Sentimientos por léxico',650,160,kind='donut');r.tablevisual('Temas',list(r.tables['Temas']),'Temas LDA',30,480,600,350);r.tablevisual('Bigramas',list(r.tables['Bigramas']),'Bigramas frecuentes',650,480,600,350)
r.page('ab','06 · Comparación A/B de canales',[('Posteriores','Comparacion'),('ComparacionCanales','Comparacion')],note='Datos observacionales: asociación entre canales, sin asignación aleatoria; no demuestra causalidad. Prueba Z y posteriores Beta.')
r.chart('Posteriores','Tasa','KPI_Densidad','Distribuciones posteriores Beta',30,160,1220,330,kind='line',legend='Canal');r.tablevisual('ComparacionCanales',list(r.tables['ComparacionCanales']),'Tasas, diferencia, Z, IC95 y P(B>A)',30,520,1220,310)
r.page('calidad','07 · Perfil de calidad',[('Calidad','Campo')]);r.chart('Calidad','Campo','KPI_Nulos','Valores ausentes · muestra original completa',30,160);r.chart('Calidad','Campo','KPI_Unicos','Cardinalidad por campo',650,160);r.tablevisual('Calidad',list(r.tables['Calidad']),'Perfil de campos · sin datos de contacto',30,470,1220,360)
result=r.finish('| Streamlit | Power BI |\n|---|---|\n| Resumen | Conversaciones, resolución, tendencias |\n| Distribución | Canal/idioma/incidencia y desglose |\n| Sentimientos | Sentimiento e industria |\n| Urgencia y resultados | Urgencia, outcomes y resolución |\n| NLP | TF-IDF, LDA, bigramas y léxico sobre 500 mensajes |\n| A/B | Todos los pares de canales, Z, IC95 y posterior Beta |\n| Calidad | Nulos y cardinalidad de las 50.000 filas |\n\nLa muestra conserva el límite inicial de Streamlit: no representa los 629 MB del dataset completo. Se omiten nombres y textos completos. Los resultados A/B no prueban efecto causal.')
print(result)

