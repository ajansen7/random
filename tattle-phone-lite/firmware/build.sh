#!/bin/sh
# Build (and optionally flash) the Tattle Phone Lite firmware with arduino-cli.
#   ./build.sh                          compile only
#   ./build.sh /dev/cu.usbmodem1101     compile + flash over USB-C (batteries in, power ON)
#   ./build.sh /dev/cu.usbmodem1101 monitor    ...then open the serial monitor
# One-time setup: brew install arduino-cli
#   arduino-cli config init
#   arduino-cli config set board_manager.additional_urls https://espressif.github.io/arduino-esp32/package_esp32_index.json
#   arduino-cli core update-index && arduino-cli core install esp32:esp32
set -e
cd "$(dirname "$0")"
FQBN="esp32:esp32:esp32s3:FlashSize=8M,PSRAM=enabled,PartitionScheme=custom,CDCOnBoot=cdc,USBMode=hwcdc"
[ -f tattle_lite/config.h ] || { echo "Copy tattle_lite/config.example.h to tattle_lite/config.h and fill it in first."; exit 1; }
arduino-cli compile -b "$FQBN" --warnings all tattle_lite
if [ -n "$1" ]; then
  arduino-cli upload -b "$FQBN" -p "$1" tattle_lite
  [ "$2" = "monitor" ] && arduino-cli monitor -p "$1" -c baudrate=115200
fi
