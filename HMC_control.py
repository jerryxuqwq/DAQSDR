import time
from pyftdi.gpio import GpioAsyncController

# ==========================================
# FT232 Pin Mapping (ADBUS)
# ------------------------------------------
# Bit 0 (ADBUS 0): CLK      (Output)
# Bit 1 (ADBUS 1): DATA     (Output - MOSI/SDI)
# Bit 2 (ADBUS 2): ENABLE   (Output - SEN)
# Bit 3 (ADBUS 3): SCAN_RX  (Output) -> Driven by FT232
# Bit 4 (ADBUS 4): SCAN_TX  (Input)  -> Read from TX Module
# Bit 5 (ADBUS 5): Reset    (Output)
# ==========================================

# 1 = Output, 0 = Input. 
# Binary: 101111 -> Hex: 0x2F
DIRECTION_MASK = 0x2F

# ==========================================
# HMC6300 Transmitter Register Map (Rows 1-27)
# ==========================================
HMC6300_TX_REGISTERS = {
    0x01: "ROW1  - PA Bias Control (Base Voltage & Bias Current)",
    0x02: "ROW2  - Tx Output, Power Down & Power Detector",
    0x03: "ROW3  - Driver Bias & IF Mixer Control",
    0x04: "ROW4  - Component Power Down Controls",
    0x05: "ROW5  - Tripler Bias 1",
    0x06: "ROW6  - Tripler Bias 2",
    0x07: "ROW7  - IF VGA Gain & Tune",
    0x08: "ROW8  - IF VGA Bias & Upmixer Tune",
    0x09: "ROW9  - IF VGA Q Control",
    0x0A: "ROW10 - Modulation & Baseband PD Controls",
    0x0B: "ROW11 - RF VGA Gain & Bias Control",
    0x0C: "ROW12 - Upmixer Calibration",
    0x10: "ROW16 - Synth Charge Pump & LDO",
    0x11: "ROW17 - Synth Lock Detect & Dividers",
    0x12: "ROW18 - Synth VCO Enables & Channel Step",
    0x13: "ROW19 - Synth Reference Mux",
    0x14: "ROW20 - Synth Feedback Divider Code",
    0x15: "ROW21 - Synth VCO Bias Trim",
    0x16: "ROW22 - Synth VCO Band Select",
    0x17: "ROW23 - Synth CP Bias Trim & VCO Offset",
    0x18: "ROW24 - Lock Detect & VCO Amplitude (Read Only)",
    0x19: "ROW25 - VCO Amplitude Flash P (Read Only)",
    0x1A: "ROW26 - VCO Amplitude Flash N (Read Only)",
    0x1B: "ROW27 - Temperature Sensor (Read Only)"
}

class HMC6300_FT232:
    def __init__(self, ftdi_url='ftdi://ftdi:232/1'):
        """Initializes the FT232 in Asynchronous Bit-Bang mode."""
        self.gpio = GpioAsyncController()
        self.gpio.configure(ftdi_url, direction=DIRECTION_MASK)
        
        # Initial states for our pins
        self.clk = 0
        self.data = 0
        self.enable = 0
        self.scan_rx = 0   # Now an output
        self.reset_pin = 1 # Assuming active-low reset; start high
        
        self._update_pins()

    def _update_pins(self):
        """Pushes the current state of all output pins to the FT232."""
        val = (self.clk << 0) | \
              (self.data << 1) | \
              (self.enable << 2) | \
              (self.scan_rx << 3) | \
              (self.reset_pin << 5)
        self.gpio.write(val)

    def _read_scan_tx(self):
        """Reads the state of ADBUS 4 (SCAN_TX) coming from the module."""
        pins = self.gpio.read()
        return (pins >> 4) & 1

    def hardware_reset(self):
        """Toggles the reset pin to reinitialize the HMC6300."""
        print("Asserting hardware reset...")
        self.reset_pin = 0
        self._update_pins()
        time.sleep(0.01) # 10ms reset pulse
        
        self.reset_pin = 1
        self._update_pins()
        time.sleep(0.01) # Wait for chip to wake up
        print("Reset complete.")

    def write_register(self, address, data, verbose=True):
        """Writes an 8-bit data word to a 6-bit register address."""
        rw_bit = 1            # 1 for Write
        chip_addr = 0b110     # Tx Chip Address
        
        # Construct the 18-bit payload (LSB first)
        payload = (data & 0xFF) | \
                  ((address & 0x3F) << 8) | \
                  ((rw_bit & 0x1) << 14) | \
                  ((chip_addr & 0x7) << 15)
                  
        # Start transaction: Enable HIGH
        self.enable = 1
        self._update_pins()
        
        # Clock out 18 bits, LSB first
        for i in range(18):
            self.data = (payload >> i) & 1
            self._update_pins()
            
            # Clock High (Data latches on rising edge)
            self.clk = 1
            self._update_pins()
            
            # Clock Low
            self.clk = 0
            self._update_pins()
            
        # End transaction: Enable LOW
        self.enable = 0
        self._update_pins()
        if verbose:
            print(f"Wrote {hex(data)} to register {hex(address)}")

    def read_register(self, address, verbose=True):
        """Reads an 8-bit data word from a 6-bit register address."""
        rw_bit = 0            # 0 for Read
        chip_addr = 0b110
        
        # Payload for read command
        payload = (0x00 & 0xFF) | \
                  ((address & 0x3F) << 8) | \
                  ((rw_bit & 0x1) << 14) | \
                  ((chip_addr & 0x7) << 15)
                  
        self.enable = 1
        self._update_pins()
        
        # 1. Clock out the 18-bit Read Command
        for i in range(18):
            self.data = (payload >> i) & 1
            self._update_pins()
            
            self.clk = 1
            self._update_pins()
            self.clk = 0
            self._update_pins()
            
        # 2. Clock in the 8-bit Data (LSB first)
        read_data = 0
        for i in range(8):
            self.clk = 1
            self._update_pins()
            
            # Read SCAN_TX (Bit 4) while clock is high
            bit = self._read_scan_tx()
            read_data |= (bit << i)
            
            self.clk = 0
            self._update_pins()
            
        self.enable = 0
        self._update_pins()
        
        if verbose:
            print(f"Read {hex(read_data)} from register {hex(address)}")
        return read_data

    def dump_all_registers(self):
        """Iterates through the mapped datasheet registers, reads them, and prints a formatted table."""
        print("\n" + "="*85)
        print(f"{'Address':<10} | {'Hex Value':<10} | {'Binary Value':<12} | {'Register Function'}")
        print("-" * 85)
        
        for address, description in HMC6300_TX_REGISTERS.items():
            # Read the register (silencing the standard print statement)
            val = self.read_register(address, verbose=False)
            
            # Format outputs
            addr_str = f"0x{address:02X}"
            hex_val = f"0x{val:02X}"
            bin_val = f"0b{val:08b}"
            
            print(f"{addr_str:<10} | {hex_val:<10} | {bin_val:<12} | {description}")
            
        print("="*85 + "\n")

    def close(self):
        """Safely release the FT232 interface."""
        self.enable = 0
        self.clk = 0
        self.data = 0
        self.scan_rx = 0
        self._update_pins()
        self.gpio.close()

# ==========================================
# Main Execution Block
# ==========================================
if __name__ == "__main__":
    # Adjust URL as necessary for your FT232 setup
    hmc = HMC6300_FT232(ftdi_url='ftdi://ftdi:2232:1:e/2')
    
    try:
        # 1. Reset the chip first
        hmc.hardware_reset()
        
        # 2. Write Example (Setting Synth Feedback Divider Code to 60GHz approx)
        # hmc.write_register(0x14, 0x18) 
        
        # 3. Read specific register example
        # val = hmc.read_register(0x05)
        
        # 4. Dump all Transmitter Registers to screen
        hmc.dump_all_registers()
        
    finally:
        hmc.close()