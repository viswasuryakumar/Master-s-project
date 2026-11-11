import React from 'react';
export type Recommendation={title:string;summary:string;basis:string;origin:string;issues:{category:string;title:string;summary:string;checks:string[];evidence:{line:number;text:string}[]}[]};
export default function RecommendedAction({action}:{action?:Recommendation|null}){
 if(!action)return null;
 return <div className="recommended-action"><strong>{action.title}</strong><p>{action.summary}</p>{action.issues.length>0?<details><summary>Checks & evidence</summary><p className="action-basis">{action.basis}</p>{action.issues.map(issue=><section key={issue.category}><h4>{issue.title}</h4><ol>{issue.checks.map(check=><li key={check}>{check}</li>)}</ol><div className="action-evidence">{issue.evidence.map(e=><code key={e.line}>[L{e.line}] {e.text}</code>)}</div></section>)}</details>:<small>{action.basis}</small>}</div>
}
