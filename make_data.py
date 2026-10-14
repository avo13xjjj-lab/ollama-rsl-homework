"""Synthetic, deterministic, template-separated dataset with explicit labels."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def generate():
    splits={
      'train': [
        '{p}从北京出发去{l}{e}，预算{b}元，日期是{d}。',
        '{p}原定去天津，后来改为去{l}{e}，日期{d}，预算{b}元。',
        '{p}去{l}{e}。旁人的出差预算为900元；{p}没有提供自己的日期和预算。',
      ],
      'validation': [
        '行程单：{p}；起点上海；活动地址{l}；事项{e}；日期{d}；可用资金{b}元。',
        '更正通知：{p}的活动地点不再是武汉，最终为{l}，{d}{e}，预算{b}元。',
        '通知：{p}将赴{l}{e}，时间待定，预算待定。附注：另一人的预算900元。',
      ],
      'test': [
        '由成都启程的{p}将在{d}于{l}{e}，本次预算{b}元。',
        '请作废此前的广州安排。{p}最终确认在{d}前往{l}{e}，预算{b}元。',
        '已知{p}的任务是在{l}{e}。尚未确定日期或预算，其他项目预算900元不属于本活动。',
      ]}
    people=['张三','李四','王五','赵六','陈七','刘八','周九','吴十','郑一','孙二']
    locations=['杭州','南京','苏州','厦门','青岛','长沙','合肥','宁波','昆明','福州']
    events=['参加会议','参加培训','参观展览','拜访客户','参加讲座']
    for split,templates in splits.items():
        items=[]
        repeats=30 if split=='train' else 10
        for category,template in enumerate(templates):
            for i in range(repeats):
                p=people[i%10]; l=locations[(i+category)%10]; e=events[i%5]
                d=f'2026-11-{i%28+1:02d}'; b=1500+i*100
                expected=dict(person=p,date=d,location=l,event=e,budget=b)
                if category==2:
                    expected.update(date=None,budget=None)
                items.append(dict(id=f'{split}_{category}_{i:02d}',category=['destination','correction','missing'][category],
                    text=template.format(p=p,l=l,e=e,d=d,b=b),expected=expected))
        path=ROOT/'data'/f'{split}.json'; path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(items,ensure_ascii=False,indent=2),encoding='utf-8')
    print('已生成90条学习、30条验证、30条测试数据；三个集合使用不同表达模板。')

if __name__=='__main__': generate()
