"""Extract source workbooks to CSV. Requires openpyxl and xlrd. Run from any directory."""
from pathlib import Path
import csv,re,hashlib,math
import xlrd,openpyxl
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'05_Model_Ready_Data'
def save(name,rows):
 if not rows:return
 with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def num(v):
 if isinstance(v,(int,float)):return float(v)
 try:return float(str(v).strip().replace(',','.'))
 except ValueError:return None
index={r['file']:r['url'] for r in csv.DictReader((ROOT/'SOURCE_INDEX.csv').open(encoding='utf-8-sig'))}
def source(p):return index.get(str(p.relative_to(ROOT)),'')
# Each annual December bulletin contains January-December monthly changes.
records=[];broad=[];checks=[]
for year in range(2016,2027):
 pattern=f'*{year}_January-December*.xls' if year<2026 else '*2026_09*.xls'
 p=next((ROOT/'04_Price_Bulletins').glob(pattern));b=xlrd.open_workbook(p)
 candidates=[s for s in b.sheets() if s.name.strip() in ['ay','əvvəlki aya']]
 assert len(candidates)==1,(year,b.sheet_names())
 s=candidates[0];header=' '.join(str(v) for i in range(4) for v in s.row_values(i))
 assert 'əvvəlki aya' in header,(year,header)
 assert str(s.cell_value(3,2)).strip().lower()=='yanvar'
 maxmonth=12 if year<2026 else 9
 labels={4:'national',5:'food_beverages_tobacco',7:'nonfood_goods',9:'services'}
 for r in range(4,s.nrows):
  label=' '.join(str(s.cell_value(r,1)).split())
  if not label:continue
  for m in range(1,maxmonth+1):
   if m+1>=s.ncols:continue
   v=num(s.cell_value(r,m+1))
   if v is None or not 50<v<200:continue
   row={'date':f'{year}-{m:02d}-01','category_label_az':label,'previous_month_100':v,'mom_inflation_pct':round(v-100,8),'source_file':str(p.relative_to(ROOT)),'source_sheet':s.name,'source_excel_row':r+1,'source_excel_column':m+2,'source_url':source(p)}
   records.append(row)
   if r in labels:broad.append(dict(date=row['date'],series=labels[r],mom_inflation_pct=row['mom_inflation_pct'],previous_month_100=v,source_label=label,source_file=row['source_file'],source_sheet=s.name,source_excel_row=r+1,source_url=source(p)))
 for r,code in labels.items():checks.append({'year':year,'series':code,'original_label':s.cell_value(r,1),'source_file':str(p.relative_to(ROOT))})
# Broad labels deliberately retain food+beverages+tobacco and exclude overlapping aggregates.
for code in ['national','food_beverages_tobacco','nonfood_goods','services']:
 rr=sorted([r for r in broad if r['series']==code],key=lambda r:r['date']);assert len(rr)==129,(code,len(rr))
 level=100.0
 for i,r in enumerate(rr):
  level*=1+r['mom_inflation_pct']/100;r['chained_index_dec2015_100']=round(level,8)
  r['yoy_inflation_pct']=round((math.prod(1+x['mom_inflation_pct']/100 for x in rr[i-11:i+1])-1)*100,8) if i>=11 else ''
# add derived fields to every row in the same field order
broad=sorted(broad,key=lambda r:(r['series'],r['date']))
save('monthly_broad_CPI_2016-01_to_2026-09.csv',broad)
save('monthly_detailed_CPI_original_labels_2016-01_to_2026-09.csv',records)
save('broad_category_label_checks.csv',checks)
# Preserve separately reported official CPI levels, rather than splicing bases.
p=next((ROOT/'02_Prices_and_Inflation').glob('Monthly_CPI*'));s=openpyxl.load_workbook(p,data_only=True).active
romans=['I','II','III','IV','V','VI','VII','VIII','IX','X','XI','XII'];levels=[];year=None
for r,row in enumerate(s.values,1):
 v=row[1]
 if isinstance(v,(int,float)) and 1900<v<2100:year=int(v)
 elif v in romans and year:
  for col,code in enumerate(['national','food_beverages_tobacco','nonfood_goods','services'],2):levels.append({'date':f'{year}-{romans.index(v)+1:02d}-01','series':code,'official_CPI_2010_100':row[col],'source_file':str(p.relative_to(ROOT)),'source_excel_row':r,'source_url':source(p)})
save('official_monthly_CPI_levels_2019-2025.csv',levels)
# National annual inflation, no interpolation.
p=next((ROOT/'02_Prices_and_Inflation').glob('National_CPI*'));s=openpyxl.load_workbook(p,data_only=True).active;annual=[]
for r,row in enumerate(s.values,1):
 if isinstance(row[1],(int,float)) and 1990<=row[1]<=2026:
  for col,code in enumerate(['national','food_beverages_tobacco','nonfood_goods','services'],2):
   v=num(row[col])
   if v:annual.append({'year':int(row[1]),'series':code,'annual_average_inflation_pct':round(v-100,8),'source_file':str(p.relative_to(ROOT)),'source_excel_row':r,'source_url':source(p)})
save('annual_broad_inflation_1993-2025.csv',annual)
# Household historical workbooks: national block only; values per capita per month.
h=[]
for topic in ['Income_per_person','Consumption_per_person']:
 p=next((ROOT/'01_Household_Income_and_Spending').glob(topic+'*'));s=xlrd.open_workbook(p).sheet_by_index(0)
 years=[(c,int(s.cell_value(4,c))) for c in range(2,s.ncols) if isinstance(s.cell_value(4,c),(int,float)) and 2000<=s.cell_value(4,c)<=2026]
 for r in range(5,s.nrows):
  label=str(s.cell_value(r,1)).strip()
  if r>5 and (any('urban' in str(v).lower() or 'rural' in str(v).lower() for v in s.row_values(r)) or any(isinstance(v,(int,float)) and 2000<=v<=2026 for v in s.row_values(r)[2:])):break
  for c,y in years:
   v=num(s.cell_value(r,c))
   if v is not None:h.append({'year':y,'group':'national','measure':topic,'category_label_en':label,'value_AZN_per_person_per_month':v,'source_file':str(p.relative_to(ROOT)),'source_excel_row':r+1,'source_url':source(p)})
save('household_national_income_spending_selected_years.csv',h)
# Expenditure weights, with nested alcohol row explicitly marked.
w=[]
for pattern,offset,groups in [('Spending_Shares*',0,['national','urban','rural']),('Spending_Expenditure_Quintiles*',20,['expenditure_Q1','expenditure_Q2','expenditure_Q3','expenditure_Q4','expenditure_Q5'])]:
 p=next((ROOT/'01_Household_Income_and_Spending').glob(pattern));s=xlrd.open_workbook(p).sheet_by_index(0)
 for r in range(7+offset,20+offset):
  label=str(s.cell_value(r,1)).strip()
  for c,g in enumerate(groups,2):
   v=num(s.cell_value(r,c))
   if v is not None:w.append({'weight_year':2024,'group':g,'category_label_en':label,'share_pct':v,'is_nested_component':int(r==8+offset),'source_file':str(p.relative_to(ROOT)),'source_excel_row':r+1,'source_url':source(p)})
save('household_spending_weights_2024.csv',w)
for g in sorted(set(r['group'] for r in w)):
 total=sum(r['share_pct'] for r in w if r['group']==g and not r['is_nested_component']);assert abs(total-100)<=0.3,(g,total)
# Structural validation and chronological split labels.
splits=[]
for r in broad:
 d=r['date'];split='train' if d<'2024-10-01' else 'validation' if d<'2025-10-01' else 'test'
 splits.append({'date':d,'series':r['series'],'mom_inflation_pct':r['mom_inflation_pct'],'split':split})
save('forecast_inputs_with_chronological_split.csv',splits)
print('Monthly broad rows',len(broad),'Detailed rows',len(records),'Annual rows',len(annual),'Household rows',len(h),'Weight rows',len(w))
