# Copyright (c) 2026 Relax Authors. All Rights Reserved.


def strip_param_name_prefix(name: str):
    prefix = "module."
    while name.startswith(prefix):
        name = name.removeprefix(prefix)
    return name
