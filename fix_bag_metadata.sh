#!/bin/bash
BAG_PATH="$1/metadata.yaml"
if [ -z "$1" ]; then echo "Usage: $0 /path/to/bag_folder"; exit 1; fi

python3 << PYEOF
import re
path = '$BAG_PATH'
with open(path) as f:
    content = f.read()

content = re.sub(r'offered_qos_profiles:\n(?:          .*\n)+', 'offered_qos_profiles: ""\n', content)
content = content.replace('offered_qos_profiles:\n          []', 'offered_qos_profiles: ""')
content = re.sub(r'\s+type_description_hash:.*\n', '\n', content)

with open(path, 'w') as f:
    f.write(content)

import yaml
yaml.safe_load(open(path))
print("Done ✓")
PYEOF
