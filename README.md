#pynfc15693demod 
Radio demodualtion and frame decoding for NFC-15693 rfid norm.

pynfc15693demod requires raw 32 bits float I/Q samples as input file, without any header.
Nominal sdr parameters:
    - Sample frequency 2MHz
    - center frequency 13.56MHz


Tested with a SDR-Play sdr at 2MHz sample frequeny and centered on 13.56MHz.

Only the VCD to VICC demodulation and frame decoding is functional. 
TODO: VICC to VCD answer demodulation and frame decoder.



**VCD** : Vicinity coupling device (eg a smartphone)
**VICC**: Vicinity integrated circuit card (eg nfc tag)
