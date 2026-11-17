#!/usr/bin/env python3
"""Generate a standalone canvas viewer of a host DAG prefix."""
import argparse
import collections
import json
import math
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--fraction', type=float, default=0.1)
    a = p.parse_args()
    if not 0 < a.fraction <= 1:
        p.error('--fraction must be in (0, 1]')
    source = json.loads(a.input.read_text())['nodes']
    selected = source[:math.ceil(len(source) * a.fraction)]
    ids = {n['id']: i for i, n in enumerate(selected)}
    nodes = []
    edges = []
    outgoing = collections.defaultdict(list)
    incoming = collections.Counter()
    for i, n in enumerate(selected):
        comm = n['name'].startswith(('Hccl', 'hcom', 'c10d::', '_c10d'))
        nodes.append(dict(id=n['id'], name=n['name'], comm=comm,
                          ts=n.get('start_ts'), duration=n.get('duration'),
                          deps=n.get('data_deps', []), outside=0))
        for dep in n.get('data_deps', []):
            if dep in ids:
                u = ids[dep]
                edges.append([u, i])
                outgoing[u].append(i)
                incoming[i] += 1
            else:
                nodes[-1]['outside'] += 1
    q = collections.deque(i for i in range(len(nodes)) if not incoming[i])
    depth = [0] * len(nodes)
    count = 0
    while q:
        u = q.popleft()
        count += 1
        for v in outgoing[u]:
            depth[v] = max(depth[v], depth[u] + 1)
            incoming[v] -= 1
            if not incoming[v]:
                q.append(v)
    if count != len(nodes):
        raise ValueError('Selected graph contains a cycle')
    connected = {i for edge in edges for i in edge}
    layers = collections.defaultdict(list)
    for i in connected:
        layers[depth[i]].append(i)
    # Keep ordering deterministic. Layer coordinates express dependencies, not time.
    for level, group in layers.items():
        group.sort(key=lambda i: nodes[i]['id'])
        for row, i in enumerate(group):
            nodes[i].update(x=level * 90, y=(row - len(group)/2) * 14, isolated=False)
    isolated = [i for i in range(len(nodes)) if i not in connected]
    for row, i in enumerate(isolated):
        nodes[i].update(x=(row % 150) * 18, y=-(max(map(len, layers.values()), default=0)/2)*14-300-(row//150)*18, isolated=True)
    data = dict(nodes=nodes, edges=edges, total=len(source), isolated=len(isolated),
                external=sum(n['outside'] for n in nodes))
    html = TEMPLATE.replace('__DATA__', json.dumps(data, ensure_ascii=False).replace('<', '\\u003c'))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(html, encoding='utf-8')
    print(f'nodes={len(nodes)} edges={len(edges)} isolated={len(isolated)} external_dependencies={data["external"]} output={a.output}')


TEMPLATE = r'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><title>Host DAG 通信与计算依赖</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#0b1220;color:#dbe7f7;font:14px system-ui}header{padding:16px 22px;background:#121f33;border-bottom:1px solid #30405a}h1{font-size:21px;margin:0 0 8px}button,input{background:#223650;color:#edf4ff;border:1px solid #47617e;border-radius:5px;padding:7px}button{cursor:pointer}label{margin-right:14px}#bar{padding:10px 22px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}#wrap{position:relative;height:calc(100vh - 170px);min-height:400px}canvas{width:100%;height:100%;display:block;touch-action:none}aside{position:absolute;right:14px;top:14px;width:310px;max-height:85%;overflow:auto;background:#142238ed;border:1px solid #3d5471;border-radius:8px;padding:15px;white-space:pre-wrap;word-break:break-word}#stats{color:#a8bed8}.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px}small{color:#a8bed8}#tip{position:absolute;bottom:10px;left:20px;color:#b5c7de}
</style>
<header><h1>Host DAG · 通信与计算依赖</h1><div id="stats"></div><small>原文件顺序的前 1/10 节点；仅展示选中节点间的 data_deps。箭头：生产者 → 消费者。横轴为拓扑层级，不是时间轴。</small></header>
<div id="bar"><button id="fit">适应画布</button><button id="zoom">放大</button><label><input id="commOnly" type="checkbox">仅通信相关边</label><label><input id="isolated" type="checkbox">显示孤立节点</label><input id="search" placeholder="输入节点 ID"><button id="find">定位</button><span><i class="dot" style="background:#ffb454"></i>通信</span><span><i class="dot" style="background:#55b7f5"></i>其他 / 计算</span><span style="color:#f4a648">通信相关边</span><span style="color:#718ba9">计算相关边</span></div>
<div id="wrap"><canvas id="canvas"></canvas><aside id="info">滚轮缩放，拖动画布，点击节点查看详情。

默认隐藏无内部依赖边的孤立节点。通信节点按 Hccl / hcom / c10d:: / _c10d 前缀分类；其余节点包含计算、视图与辅助操作。

此图呈现文件中的推导依赖，不表示已验证的设备同步关系。</aside><div id="tip"></div></div>
<script>
const D=__DATA__, N=D.nodes,E=D.edges,c=document.querySelector('canvas'),ctx=c.getContext('2d');
const info=document.querySelector('#info'),tip=document.querySelector('#tip');
let scale=1,ox=0,oy=0,w=0,h=0,selected=-1,pending=false,drag=null,moved=false;
const adjacency=N.map(()=>[]);E.forEach(([u,v])=>{adjacency[u].push(v);adjacency[v].push(u)});
const types=[0,0,0,0];E.forEach(([u,v])=>types[(N[u].comm?2:0)+(N[v].comm?1:0)]++);
document.querySelector('#stats').textContent=`展示 ${N.length.toLocaleString()} / ${D.total.toLocaleString()} 节点 · ${E.length.toLocaleString()} 条内部边 · 通信→其他 ${types[2]} · 其他→通信 ${types[1]} · 通信→通信 ${types[3]} · 孤立 ${D.isolated} · 区域外前驱 ${D.external}`;
function visible(n){return (!n.isolated||document.querySelector('#isolated').checked)&&(!document.querySelector('#commOnly').checked||n.comm||adjacency[N.indexOf(n)].some(i=>N[i].comm))}
let mask=[];function filters(){const only=document.querySelector('#commOnly').checked,iso=document.querySelector('#isolated').checked;mask=N.map((n,i)=>(!n.isolated||iso)&&(!only||n.comm||adjacency[i].some(j=>N[j].comm)));request()}
function fit(){let xs=[],ys=[];N.forEach((n,i)=>{if(mask[i]){xs.push(n.x);ys.push(n.y)}});if(!xs.length)return;let minx=Math.min(...xs),maxx=Math.max(...xs),miny=Math.min(...ys),maxy=Math.max(...ys);scale=Math.min((w-370)/(maxx-minx+80),(h-70)/(maxy-miny+80));ox=40-minx*scale;oy=h/2-(miny+maxy)/2*scale;request()}
function request(){if(!pending){pending=true;requestAnimationFrame(draw)}}
function draw(){pending=false;ctx.clearRect(0,0,w,h);const only=document.querySelector('#commOnly').checked;const near=new Set(selected<0?[]:adjacency[selected]);
E.forEach(([u,v])=>{if(!mask[u]||!mask[v]||(only&&!N[u].comm&&!N[v].comm))return;let a=N[u],b=N[v],x=ox+a.x*scale,y=oy+a.y*scale,X=ox+b.x*scale,Y=oy+b.y*scale;if(Math.max(x,X)<0||Math.min(x,X)>w||Math.max(y,Y)<0||Math.min(y,Y)>h)return;let active=u===selected||v===selected;ctx.strokeStyle=active?'#e9fb84':(a.comm||b.comm?'#f4a648':'#718ba9');ctx.globalAlpha=active?1:selected>=0?.07:.28;ctx.lineWidth=active?2:.6;ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(X,Y);ctx.stroke();if(active||scale>.5){let angle=Math.atan2(Y-y,X-x),r=4;ctx.beginPath();ctx.moveTo(X,Y);ctx.lineTo(X-r*Math.cos(angle-.45),Y-r*Math.sin(angle-.45));ctx.lineTo(X-r*Math.cos(angle+.45),Y-r*Math.sin(angle+.45));ctx.closePath();ctx.fillStyle=ctx.strokeStyle;ctx.fill()}});
N.forEach((n,i)=>{if(!mask[i])return;let x=ox+n.x*scale,y=oy+n.y*scale;if(x< -10||x>w+10||y< -10||y>h+10)return;ctx.globalAlpha=selected<0||i===selected||near.has(i)?1:.2;ctx.fillStyle=i===selected?'#e9fb84':n.comm?'#ffb454':'#55b7f5';ctx.beginPath();ctx.arc(x,y,i===selected?6:Math.max(1.1,Math.min(4,scale*3)),0,Math.PI*2);ctx.fill();if(scale>1.6&&(n.comm||i===selected)){ctx.font='11px system-ui';ctx.fillText(`${n.id} ${n.name}`,x+7,y-5)}});ctx.globalAlpha=1;tip.textContent=`缩放 ${(scale*100).toFixed(1)}% · 已选节点 ${selected<0?'无':N[selected].id}`}
function choose(i){selected=i;let n=N[i];info.textContent=`节点 ${n.id}\n${n.name}\n\n类别：${n.comm?'通信':'其他 / 计算'}\n开始时间(us)：${n.ts??'缺失'}\n持续时间(us)：${n.duration??'缺失'}\n\n前驱 ID：${n.deps.join(', ')||'无'}\n区域外前驱：${n.outside}\n内部后继 ID：${E.filter(e=>e[0]===i).map(e=>N[e[1]].id).join(', ')||'无'}`;request()}
c.onwheel=e=>{e.preventDefault();let r=c.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top,f=Math.exp(-e.deltaY*.001);let next=Math.max(.0001,Math.min(8,scale*f));f=next/scale;ox=x-(x-ox)*f;oy=y-(y-oy)*f;scale=next;request()};
c.onpointerdown=e=>{drag=[e.clientX,e.clientY,ox,oy];moved=false;c.setPointerCapture(e.pointerId)};
c.onpointermove=e=>{if(drag){let dx=e.clientX-drag[0],dy=e.clientY-drag[1];if(Math.abs(dx)+Math.abs(dy)>4)moved=true;ox=drag[2]+dx;oy=drag[3]+dy;request()}};
c.onpointerup=e=>{drag=null;if(moved)return;let r=c.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top,best=-1,dist=100;N.forEach((n,i)=>{if(mask[i]){let d=(ox+n.x*scale-x)**2+(oy+n.y*scale-y)**2;if(d<dist){dist=d;best=i}}});if(best>=0)choose(best)};
document.querySelector('#fit').onclick=fit;document.querySelector('#zoom').onclick=()=>{ox=w/2+(ox-w/2)*2;oy=h/2+(oy-h/2)*2;scale*=2;request()};
document.querySelector('#commOnly').onchange=()=>{filters();fit()};document.querySelector('#isolated').onchange=()=>{filters();fit()};
document.querySelector('#find').onclick=()=>{let id=Number(document.querySelector('#search').value),i=N.findIndex(n=>n.id===id);if(i<0){info.textContent='该 ID 不在本次展示范围内。';return}document.querySelector('#commOnly').checked=false;if(N[i].isolated)document.querySelector('#isolated').checked=true;filters();scale=1.5;ox=(w-330)/2-N[i].x*scale;oy=h/2-N[i].y*scale;choose(i)};
function resize(){const r=c.getBoundingClientRect();w=r.width;h=r.height;c.width=w*devicePixelRatio;c.height=h*devicePixelRatio;ctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);request()}window.onresize=resize;resize();filters();fit();
</script></html>'''

if __name__ == '__main__':
    main()
