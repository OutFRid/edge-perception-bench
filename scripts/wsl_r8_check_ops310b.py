#!/usr/bin/env python3
"""R8 诊断⑩：310B 的 AscendDequant/AscendQuant 算子支持清单。"""
import json

d = json.load(open("/root/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/config/ascend310b/aic-ascend310b-ops-info.json"))
for k, v in d.items():
    if "AscendQuant" in k:
        print(k)
        print("  full:", json.dumps(v, ensure_ascii=False))
        print()
