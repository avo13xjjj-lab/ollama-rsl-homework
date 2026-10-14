"""Local experience-learning RSL. Python standard library only."""
import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIELDS = ('person', 'date', 'location', 'event', 'budget')
BASE = '''从中文文本中提取一个最终有效的活动，输出 JSON，必须且只能包含 person,date,location,event,budget。
person 为人物姓名；date 为 YYYY-MM-DD；location 为活动地点；event 为原文活动动词及宾语，例如参加会议；budget 为数字金额（元）。
缺失或无法确定的字段为 null，不猜测。不输出解释。'''

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

class Client:
    def __init__(self, model='gpt-oss:20b', url='http://127.0.0.1:11434'):
        self.model, self.url = model, url.rstrip('/')

    def call(self, system, user):
        payload = dict(model=self.model, messages=[dict(role='system', content=system),
                       dict(role='user', content=user)], stream=False, think='low', format='json',
                       options=dict(num_ctx=4096, num_predict=2048, temperature=0, seed=42))
        req = urllib.request.Request(self.url+'/api/chat',
              data=json.dumps(payload).encode(), headers={'Content-Type':'application/json'})
        start = time.perf_counter()
        with urllib.request.urlopen(req, timeout=600) as response:
            result = json.load(response)
        raw = result['message'].get('content', '')
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        return dict(raw=raw, parsed=parsed, seconds=round(time.perf_counter()-start, 3),
                    tokens=result.get('eval_count'), done_reason=result.get('done_reason'))

def errors(answer, expected):
    if not isinstance(answer, dict):
        return ['回答不是合法 JSON 对象']
    result = []
    if set(answer) != set(FIELDS):
        result.append('字段必须且只能包含 '+','.join(FIELDS))
    for field in FIELDS:
        actual = answer.get(field)
        target = expected[field]
        if field == 'budget':
            correct = (actual is None if target is None else
                       isinstance(actual, (int, float)) and not isinstance(actual, bool) and actual == target)
        else:
            correct = field in answer and actual == target
        if not correct:
            result.append(f'{field}: 实际 {actual!r}，标准 {target!r}')
    return result

def prompt(memory):
    rules = '\n'.join(f'{i+1}. {r["rule"]}' for i,r in enumerate(memory))
    return BASE + ('\n经过验证的经验：\n'+rules if rules else '')

def predict(client, item, memory):
    return client.call(prompt(memory), item['text'])

def measure(client, dataset, memory):
    records = []
    for item in dataset:
        response = predict(client, item, memory)
        issues = errors(response['parsed'], item['expected'])
        records.append(dict(id=item['id'], text=item['text'], expected=item['expected'],
                            response=response, errors=issues, correct=not issues))
    n = len(records)
    fields_ok = 0
    for r in records:
        answer = r['response']['parsed']
        if isinstance(answer, dict):
            for field in FIELDS:
                if field in answer:
                    fields_ok += not errors({**r['expected'], field:answer[field]}, r['expected'])
    return dict(count=n, exact_accuracy=sum(r['correct'] for r in records)/n if n else 0,
                field_accuracy=fields_ok/(n*len(FIELDS)) if n else 0,
                json_object_rate=sum(isinstance(r['response']['parsed'],dict) for r in records)/n if n else 0,
                mean_seconds=sum(r['response']['seconds'] for r in records)/n if n else 0,
                records=records)

def learn(client, rounds, limit):
    train, validation = read(ROOT/'data/train.json'), read(ROOT/'data/validation.json')
    if limit:
        step = max(1, len(train)//limit)
        train = train[::step][:limit]
    memory = []
    history = []
    run = ROOT/'results/learning'
    save(run/'run_config.json', dict(model=client.model, url=client.url, rounds=rounds,
         learning_samples=len(train), validation_samples=len(validation), num_ctx=4096,
         temperature=0, seed=42, think='low', max_repairs=2, learning_target='experience rules'))
    save(run/'memory_v0.json', memory)
    # Validation is the same fixed set for every candidate; no test answers are used.
    current = measure(client, validation, memory)
    for round_id in range(1, rounds+1):
        first_correct = 0
        for item in train:
            first = predict(client, item, memory)
            issues = errors(first['parsed'], item['expected'])
            first_correct += not issues
            entry = dict(round=round_id, id=item['id'], first=first, errors=issues, repairs=[])
            candidate = None
            if issues:
                last = first
                for attempt in range(2):
                    last = client.call(prompt(memory), json.dumps(dict(text=item['text'],
                          previous=last['raw'], feedback=issues,
                          instruction='按反馈修正，输出完整 JSON。'), ensure_ascii=False))
                    issues = errors(last['parsed'], item['expected'])
                    entry['repairs'].append(dict(response=last, errors=issues))
                    if not issues:
                        break
                if not issues and len(memory) < 15:
                    generated = client.call('根据错误案例提炼一条可泛化的信息抽取规则。返回 {"rule":"规则"}。规则不包含案例人名、地点、具体日期或金额，不改变任务字段定义。',
                        json.dumps(dict(text=item['text'], wrong=first['parsed'], correct=item['expected'],
                                        feedback=entry['errors']), ensure_ascii=False))
                    obj = generated['parsed']
                    rule = obj.get('rule') if isinstance(obj,dict) else None
                    if isinstance(rule,str) and 0 < len(rule) <= 400 and all(rule != r['rule'] for r in memory):
                        candidate = dict(rule=rule, source=item['id'], learned_round=round_id)
                        trial = measure(client, validation, memory+[candidate])
                        no_regression = all(not old['correct'] or new['correct']
                                            for old,new in zip(current['records'],trial['records']))
                        accepted = no_regression and trial['exact_accuracy'] > current['exact_accuracy']
                        entry['candidate'] = dict(rule=rule, accepted=accepted,
                             before=current['exact_accuracy'], after=trial['exact_accuracy'],
                             no_regression=no_regression)
                        save(run/f'candidate_{round_id}_{item["id"]}.json', trial)
                        if accepted:
                            memory.append(candidate)
                            current = trial
            history.append(entry)
            save(run/'history.json', history)
            save(ROOT/'memory.json', memory)
            print(f'轮次 {round_id} {item["id"]} 首答={not entry["errors"]} 经验={len(memory)}', flush=True)
        save(run/f'memory_v{round_id}.json', memory)
        save(run/f'round_{round_id}.json', dict(first_accuracy=first_correct/len(train),
              memory_count=len(memory), validation_accuracy=current['exact_accuracy']))
    return memory

def report(results, memory):
    lines = ['# RSL 实验报告', '', '学习对象：经验规则；GPT-OSS 模型权重未更新。', '',
             '测试时不向模型提供标准答案、不执行基于标准答案的修正。', '',
             '| 方法 | 样本数 | 完全正确率 | 字段准确率 | JSON对象率 | 平均秒数 |',
             '|---|---:|---:|---:|---:|---:|']
    for name, value in results.items():
        lines.append(f'| {name} | {value["count"]} | {value["exact_accuracy"]:.1%} | {value["field_accuracy"]:.1%} | {value["json_object_rate"]:.1%} | {value["mean_seconds"]:.3f} |')
    lines += ['', f'有效经验数量：{len(memory)}。', '',
              '结果来自合成数据和固定种子下的单次实验，不能代表一般任务表现。',
              '规则根据验证集筛选，验证集可能过拟合；测试集只用于最终评分。',
              '若未保留经验或未提升准确率，应如实报告，不能宣称学习必然有效。']
    if (ROOT/'results/learning/history.json').exists():
        history = read(ROOT/'results/learning/history.json')
        failed = [x for x in history if x['errors']]
        repaired = sum(bool(x['repairs']) and not x['repairs'][-1]['errors'] for x in failed)
        candidates = [x for x in history if 'candidate' in x]
        lines += ['', '## 学习闭环记录', '',
            f'共处理 {len(history)} 次学习样本；首次失败 {len(failed)} 次，其中 {repaired} 次在两次修正以内成功。',
            f'进入验证的候选经验 {len(candidates)} 条，接受 {sum(x["candidate"]["accepted"] for x in candidates)} 条。', '',
            '| 学习轮次 | 首次正确率（在线更新中） | 有效经验数 | 验证正确率 |',
            '|---|---:|---:|---:|']
        for path in sorted((ROOT/'results/learning').glob('round_*.json')):
            row = read(path)
            lines.append(f'| {path.stem} | {row["first_accuracy"]:.1%} | {row["memory_count"]} | {row["validation_accuracy"]:.1%} |')
        for item in candidates:
            candidate = item['candidate']
            if candidate['accepted']:
                lines += ['', f'接受案例 `{item["id"]}`：{candidate["rule"]}',
                          f'验证集正确率由 {candidate["before"]:.1%} 变为 {candidate["after"]:.1%}。']
        unrepaired = next((x for x in failed if x['repairs'] and x['repairs'][-1]['errors']),None)
        if unrepaired:
            lines += ['', f'修正失败示例 `{unrepaired["id"]}`：'+ '；'.join(unrepaired['errors']),
                      '两次修正仍未通过，未提炼或接受该失败案例的经验。']
        lines += ['', '在线轮次指标会受当轮经验变化及样本顺序影响，不能直接当作冻结模型的前后对比。']
    save(ROOT/'results/evaluation.json', results)
    (ROOT/'results/report.md').write_text('\n'.join(lines),encoding='utf-8')

def main():
    parser = argparse.ArgumentParser(description='Ollama 经验学习 RSL')
    parser.add_argument('command', choices=['init','chat','learn','evaluate','demo'])
    parser.add_argument('--model',default='gpt-oss:20b')
    parser.add_argument('--url',default='http://127.0.0.1:11434')
    parser.add_argument('--rounds',type=int,default=3)
    parser.add_argument('--limit',type=int,default=0,help='限制学习样本数，仅用于快速演示')
    parser.add_argument('--text',default=None)
    args = parser.parse_args()
    if args.rounds < 1 or args.limit < 0:
        parser.error('rounds 必须大于0，limit 不能为负数')
    client = Client(args.model,args.url)
    if args.command == 'init':
        from make_data import generate
        generate()
    elif args.command in ('chat','demo'):
        memory = read(ROOT/'memory.json') if (ROOT/'memory.json').exists() else []
        if args.command == 'demo' or args.text:
            text = args.text or '张三从北京出发去杭州参加会议，预算一千五百元。'
            print(json.dumps(predict(client,dict(text=text),memory),ensure_ascii=False,indent=2))
        else:
            print('输入待抽取文本，/bye 退出。每次输入独立处理。')
            while True:
                text = input('文本：').strip()
                if text == '/bye':
                    break
                if text:
                    print(json.dumps(predict(client,dict(text=text),memory),ensure_ascii=False,indent=2))
    elif args.command == 'learn':
        learn(client,args.rounds,args.limit)
    else:
        memory = read(ROOT/'memory.json') if (ROOT/'memory.json').exists() else []
        test = read(ROOT/'data/test.json')
        results = dict(baseline=measure(client,test,[]), rsl=measure(client,test,memory))
        report(results,memory)
        print('评测完成：results/report.md')

if __name__ == '__main__':
    try:
        main()
    except (urllib.error.URLError, TimeoutError) as exc:
        raise SystemExit(f'Ollama 请求失败，请确认服务和模型可用：{exc}')
