import paramiko
from scp import SCPClient
import sys


def createSSHClient(server, port, user, password):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, user, password)
    return client


# Define progress callback that prints the current percentage completed for the file
def progress(filename, size, sent):
    sys.stdout.write("%s's progress: %.2f%%   \r" % (filename, float(sent) / float(size) * 100))


if __name__ == "__main__":
    server = "192.168.1.1"
    port = 22
    user = "root"
    password = "123"

    ssh = createSSHClient(server, port, user, password)
    scp = SCPClient(ssh.get_transport(), progress=progress)
    remote_path = "~/runs"
    local_path = r""
    scp.get(remote_path=remote_path, local_path=local_path, recursive=True)
    scp.close()
