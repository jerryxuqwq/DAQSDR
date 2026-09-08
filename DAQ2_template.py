import adi

NUM_SAMPLES = int(1024*1024/4)

dev = adi.DAQ2("ip:analog.local")
dev._ctx.set_timeout(3000)
dev._rxadc.set_kernel_buffers_count(1)
dev._txdac.set_kernel_buffers_count(2)
dev.rx_enabled_channels = [0, 1]
dev.tx_enabled_channels = [0, 1]
dev.rx_buffer_size = NUM_SAMPLES
dev.tx_cyclic_buffer = False
fs = int(dev.sample_rate)
dev.dds_single_tone(fs / 10, 0.0, channel=0)
print("Device Started")


for r in range(1):
    dev.rx_sync_start = "arm"
    dev.tx_sync_start = "arm"
    if not ("arm" == dev.tx_sync_start == dev.rx_sync_start):
        raise Exception(
            "Unexpected SYNC status: TX "
            + dev.tx_sync_start
            + " RX: "
            + dev.rx_sync_start
        )
    if r == 0:
        dev.tx_destroy_buffer()
    dev.rx_destroy_buffer()
    dev._rx_init_channels()
    dev.tx(tx_iq)
    dev.tx_sync_start = "trigger_manual"

    if not ("disarm" == dev.tx_sync_start == dev.rx_sync_start):
        raise Exception(
            "Unexpected SYNC status: TX "
            + dev.tx_sync_start
            + " RX: "
            + dev.rx_sync_start
        )
    try:
        x = dev.rx()
    except Exception as e:
        print("Run#", r, " ----------------------------- FAILED:", e)
        continue