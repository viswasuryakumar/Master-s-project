import React,{useState} from 'react';
const series=[
 {key:'precision',label:'Precision',color:'#168e78'},
 {key:'recall',label:'Recall',color:'#547bc5'},
 {key:'f1',label:'F1 score',color:'#9560b7'},
 {key:'accuracy',label:'Accuracy',color:'#bd782f'},
] as const;
type Key=typeof series[number]['key'];
type Detector={id:string;name:string;metrics:({[K in Key]:number|null}&{evaluated:number})|null};
const percent=(value:number|null|undefined)=>value==null?'N/A':`${(value*100).toFixed(1)}%`;
export default function PerformanceMetrics({detectors}:{detectors:Detector[]}){
 const [selected,setSelected]=useState('all');
 const visible=series.filter(s=>selected==='all'||s.key===selected);
 const left=54,top=35,height=220,group=190,width=left+group*detectors.length+18;
 return <div className="performance-section">
  <div className="performance-cards">{detectors.map(d=><article key={d.id} aria-label={d.name+' performance'}><h3>{d.name}</h3><div className="performance-values">{series.map(s=><div key={s.key}><span><i style={{background:s.color}}/>{s.label}</span><strong>{percent(d.metrics?.[s.key])}</strong></div>)}</div><small>{d.metrics?`${d.metrics.evaluated} shared labeled sequences evaluated`:'Awaiting labeled evaluation results'}</small></article>)}</div>
  <section className="performance-chart" aria-label="Performance metric graph"><div className="performance-chart-heading"><div><h3>Detector performance</h3><p>Compare precision, recall, F1 score, and accuracy on the same evaluation set.</p></div><label>Metric<select aria-label="Chart metric" value={selected} onChange={e=>setSelected(e.target.value)}><option value="all">All metrics</option>{series.map(s=><option key={s.key} value={s.key}>{s.label}</option>)}</select></label></div>
   <div className="performance-legend">{visible.map(s=><span key={s.key}><i style={{background:s.color}}/>{s.label}</span>)}</div>
   <div className="performance-plot"><svg viewBox={`0 0 ${width} 320`} role="img" aria-label="Grouped bar chart comparing simulated detector metrics from zero to one hundred percent">
    <title>Simulated detector metric comparison</title><desc>Each group is a detector. Bars show percentages on a fixed 0–100% scale. Missing metrics are marked N/A, not zero.</desc>
    {[0,25,50,75,100].map(value=>{const y=top+height-value/100*height;return <g key={value}><line x1={left} y1={y} x2={width-18} y2={y} stroke="#dce5df" strokeDasharray={value?'4 4':undefined}/><text x={left-10} y={y+4} textAnchor="end" fontSize="11" fill="#607168">{value}%</text></g>})}
    {detectors.map((d,i)=>{const barWidth=visible.length===1?55:32,gap=8,totalWidth=visible.length*barWidth+(visible.length-1)*gap,start=left+i*group+(group-totalWidth)/2;return <g key={d.id}>{visible.map((s,j)=>{const value=d.metrics?.[s.key],barHeight=value==null?0:Math.max(0,Math.min(value,1))*height,x=start+j*(barWidth+gap),y=top+height-barHeight;return <g key={s.key} tabIndex={0} aria-label={`${d.name}, ${s.label}: ${percent(value)}`}><title>{d.name} · {s.label}: {percent(value)}</title>{value!=null&&<rect x={x} y={y} width={barWidth} height={barHeight} rx={3} fill={s.color}/>}<text x={x+barWidth/2} y={value==null?top+height-8:y-8} textAnchor="middle" fontSize="10" fill="#34483d">{percent(value)}</text></g>})}<text x={left+i*group+group/2} y={top+height+27} textAnchor="middle" fontSize="12" fontWeight="600" fill="#34483d">{d.name}</text></g>})}
   </svg></div><p className="performance-footnote">Simulated policies, not real algorithm benchmarks. N/A means the metric cannot yet be calculated.</p>
  </section>
 </div>
}
