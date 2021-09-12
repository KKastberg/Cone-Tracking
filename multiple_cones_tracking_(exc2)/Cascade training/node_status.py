from subprocess import run, PIPE

def bash(cmd: str) -> str:
    split_cmd: [] = cmd.split(" ")
    split_cmd = [txt.replace("…", " ") for txt in split_cmd]
    process = run(split_cmd, text=True, stdout=PIPE)
    return str(process.stdout)

if __name__ == '__main__':
    nodes = ["192.168.1.104", "192.168.1.58", "192.168.1.80", "192.168.1.185", "192.168.1.62"]
    types = ["", "_24x24", "_32x32", "_40x40", "_50x50"]

    status = ""
    for node in nodes:
        status += f"\n{node} has the following: \n"
        for type in types:
            top_stage = None
            try:
                ls_out = bash(f'ssh ubuntu@{node} ls…./training_deployment{type}/data')
            except: pass
            files = ls_out.split("\n")
            for file in files:
                if "stage" in file and (not top_stage or int(file[5:-4]) > int(top_stage[5:-4])):
                    top_stage = file
            if top_stage: status += f"{type:} - {top_stage}\n"
    print(status)