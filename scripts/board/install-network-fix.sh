#!/bin/bash
# One-shot installer for Atlas 200I DK A2 usb0 static-IP (192.168.137.2) persistence.
# Run ON the board as root, from inside the scripts/board/ directory:
#     sudo bash install-network-fix.sh
# It installs the resident watchdog, writes the connman blacklist, enables + starts
# the service, and immediately re-asserts 137.2. Existing configs are backed up first.
set -uo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
TS="$(date +%Y%m%d-%H%M%S)"
BAK="/root/netbak-$TS"

for f in dk-usb0-fix.sh dk-usb0-fix.service 90-usb0-net.rules; do
    [ -f "$DIR/$f" ] || { echo "ERROR: $f not found next to this script"; exit 1; }
done

echo "[1/5] backup existing network configs -> $BAK"
mkdir -p "$BAK"
for f in /etc/rc.local /etc/netplan/01-netcfg.yaml /etc/connman/main.conf /etc/systemd/network/10-usb0.network; do
    [ -f "$f" ] && cp -a "$f" "$BAK/" 2>/dev/null || true
done

echo "[2/5] install resident watchdog"
install -m 0755 "$DIR/dk-usb0-fix.sh"     /usr/local/bin/dk-usb0-fix.sh
install -m 0644 "$DIR/dk-usb0-fix.service" /etc/systemd/system/dk-usb0-fix.service
install -m 0644 "$DIR/90-usb0-net.rules"   /etc/udev/rules.d/90-usb0-net.rules

echo "[3/5] add usb0 to connman blacklist (if connman present)"
if [ -f /etc/connman/main.conf ]; then
    sed -i -E 's/^#?NetworkInterfaceBlacklist=.*/NetworkInterfaceBlacklist=vm,eth,eth1,br,bl,ran,pp,tun,wlan,usb0/' /etc/connman/main.conf
    systemctl restart connman 2>/dev/null || true
fi

echo "[4/5] enable + start service"
systemctl daemon-reload
systemctl enable --now dk-usb0-fix.service
udevadm control --reload-rules 2>/dev/null || true

echo "[5/5] immediate apply (so you do not wait for next 15s tick)"
ip addr del 192.168.0.2/24 dev usb0 2>/dev/null || true
ip addr add 192.168.137.2/24 dev usb0 2>/dev/null || true
ip link set usb0 up 2>/dev/null || true
ip route replace default via 192.168.137.1 dev usb0 2>/dev/null || true

echo
echo "==== result ===="
echo "service : $(systemctl is-active dk-usb0-fix.service) / $(systemctl is-enabled dk-usb0-fix.service)"
echo "usb0    : $(ip -4 addr show usb0 2>/dev/null | grep -o 'inet [0-9.]*' | tr '\n' ' ')"
echo "backup  : $BAK"
echo
echo "Optional cleanups (NOT auto-applied, see docs/DK网络持久化配置.md 4.2):"
echo "  - /etc/rc.local   : remove old 'ip addr add 137.2' + resolv.conf overwrite"
echo "  - netplan eth1    : drop 192.168.137.100/24 (same subnet as usb0 -> route clash)"
