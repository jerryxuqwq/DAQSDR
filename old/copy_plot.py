import paramiko
import os
import time
import glob
import numpy as np
import matplotlib.pyplot as plt

REMOTE_HOST = 'analog.local'
REMOTE_USER = 'root'
REMOTE_DIR = '/root/'
LOCAL_TMPDIR = 'data'
BIN_FILE_SIZE = 4  # Assuming float32 values, so 4 bytes per sample

fs = 1000000000  # 1 GSPS, change to your actual sample rate

# Connect via SSH and fetch the files
def fetch_files():
    if not os.path.exists(LOCAL_TMPDIR):
        os.makedirs(LOCAL_TMPDIR)

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(REMOTE_HOST, username=REMOTE_USER,password='analog')
    sftp = ssh.open_sftp()
    # Fetch tx_iq.bin
    #print(sftp.listdir(REMOTE_DIR))  # List files to ensure connection is working
    sftp.get(os.path.join(REMOTE_DIR, 'tx_iq.npy'), os.path.join(LOCAL_TMPDIR, 'tx_iq.npy'))
    # Fetch all rx_iq_*.bin files
    files = sftp.listdir(REMOTE_DIR)
    rx_files = [f for f in files if f.startswith('rx_iq_') and f.endswith('.npy')]
    local_rx_files = []
    for f in rx_files:
        remote_path = os.path.join(REMOTE_DIR, f)
        local_path = os.path.join(LOCAL_TMPDIR, f)
        sftp.get(remote_path, local_path)
        local_rx_files.append(local_path)
    sftp.close()
    ssh.close()
    return os.path.join(LOCAL_TMPDIR, 'tx_iq.npy'), local_rx_files


def main():
    
    tx_file, rx_files = fetch_files()
    x = np.load(tx_file)
    plt.plot(np.real(x), label='TX', alpha=0.7)
    for rfile in rx_files:
        r = np.load(rfile)
        plt.plot(np.real(r), label=os.path.basename(rfile), alpha=1)
    plt.legend()
    plt.title("MxFE Phase Sync @ " + str(fs / 1000000) + " MSPS")
    #plt.draw()
    plt.show()

if __name__ == '__main__':
    main()