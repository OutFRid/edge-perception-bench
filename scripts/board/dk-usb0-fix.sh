#!/bin/bash
# Enforce management IP on the Atlas 200I DK A2 RNDIS gadget interface (usb0).
# Root cause: a late one-shot vendor action (before/after connman) assigns the
# factory 192.168.0.2 to usb0 during boot, overwriting the static 192.168.137.2.
# A single boot-time apply loses that race, so this runs as a lightweight
# resident watchdog: every 15s, if 137.2 is missing, remove 0.2 and re-add 137.2.
# Self-heals across reboot, USB replug and any late/repeated setter.
IF=usb0
while true; do
    if ! ip -4 addr show dev "$IF" 2>/dev/null | grep -q '192\.168\.137\.2'; then
        ip addr del 192.168.0.2/24 dev "$IF" 2>/dev/null
        ip addr add 192.168.137.2/24 dev "$IF" 2>/dev/null
    fi
    ip link set "$IF" up 2>/dev/null
    ip route replace default via 192.168.137.1 dev "$IF" 2>/dev/null
    sleep 15
done
