# DK 网络持久化配置（usb0 静态 IP 192.168.137.2）

> 适用硬件：华为昇腾 Atlas 200I DK A2（Ascend 310B）
> 适用系统：板载 Ubuntu 22.04（davinci-mini）
> 现象：每次重启 / USB 拔插 / 睡眠断电后，管理口 `usb0` 的固定 IP `192.168.137.2` 会掉回出厂 `192.168.0.2`，导致 SSH 连不上，需要手动重新配置。本文记录根因与最终持久化方案。

---

## 1. 一句话结论

把 `192.168.137.2` 交给一个**开机自启、常驻后台、每 15 秒巡检自愈**的 systemd watchdog（`dk-usb0-fix.service`）来保证；不再依赖"开机只跑一次"的配置，从而彻底绕过厂商固件在开机后期对 `usb0` 的一次晚到改写。

---

## 2. 拓扑与命名

```
Windows PC  ──USB(RNDIS gadget)──  Atlas 200I DK A2
  192.168.137.1  (USB RNDIS6 网卡 / ICS 共享)        192.168.137.2  (usb0)
```

- 板子通过 USB 向 Windows 枚举出一个 RNDIS 网卡（Windows 侧叫 "USB RNDIS6"），板子侧对应接口是 `usb0`（gadget，driver `g_ether`）。
- Windows 用 ICS（网络共享）在该 RNDIS 网卡上充当网关 `192.168.137.1`，板子 `usb0` 设为 `192.168.137.2`，二者同网段 `192.168.137.0/24`。
- 每次开机 gadget 的 MAC 会随机化，属正常现象（会让 Windows 侧 ARP 缓存短暂失效，不影响持久化方案）。

---

## 3. 根因分析（为什么"配得好好的 IP 会掉"）

板子上**同时存在三套网络管理机制**，且有一个厂商晚到动作：

| 机制 | 谁在跑 | 对 usb0 做了什么 |
|------|--------|------------------|
| systemd-networkd | `10-usb0.network` 指定 `Address=192.168.137.2/24` | 开机早期把 usb0 设成 137.2 ✅ |
| connman (connmand) | 开机中后期托管接口 | 曾把 usb0 重新拿去做 DHCP/出厂配置 ❌ |
| 厂商晚到 setter | 开机后期**单次**把 usb0 改回出厂 `192.168.0.2` | 全盘 grep 找不到 `0.2` 字样，疑似二进制/configfs 工具 ❌ |

**关键发现（串口 debug 实测）**：重启后即使 networkd 已把 usb0 配成 137.2，开机后期仍有一个**晚到的、只发生一次**的动作把它改回 `192.168.0.2`。任何"只在开机跑一次"的方案（networkd 单跑、systemd oneshot、udev `add` 触发）都发生在这个动作**之前**，抢不过它 —— 这就是 7 月以及多次尝试失败的真正原因。

---

## 4. 最终方案

### 4.1 常驻 watchdog（核心，根治）

`scripts/board/dk-usb0-fix.sh` → 装到 `/usr/local/bin/`：每 15 秒检查一次，只要 `137.2` 不在，就删掉 `0.2`、加回 `137.2`，并保住链路 up 和默认路由。

```bash
#!/bin/bash
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
```

`scripts/board/dk-usb0-fix.service` → 装到 `/etc/systemd/system/`，`Type=simple` + `Restart=always`，`enabled` 开机自启：

```ini
[Unit]
Description=Enforce static IP 192.168.137.2 on Ascend DK usb0 gadget (resident watchdog)
After=network-online.target connman.service
Wants=network-online.target

[Service]
Type=simple
ExecStart=/usr/local/bin/dk-usb0-fix.sh
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

`scripts/board/90-usb0-net.rules` → 装到 `/etc/udev/rules.d/`，覆盖 USB 热插拔场景：

```
ACTION=="add", SUBSYSTEM=="net", KERNEL=="usb0", RUN+="/bin/systemctl --no-block start dk-usb0-fix.service"
```

### 4.2 消除冲突源（降低抖动，非必需但推荐）

这些改动在当前板子上已完成，原文件备份在板子 `/root/netbak-20261005/`：

1. **connman 黑名单**：`/etc/connman/main.conf` 取消注释并加入 `usb0`，让 connmand 不再接管管理口：
   ```
   NetworkInterfaceBlacklist=vm,eth,eth1,br,bl,ran,pp,tun,wlan,usb0
   ```
2. **rc.local 去冗余**：删掉旧的 `ip addr add 137.2 ...` 与会覆盖 `/etc/resolv.conf` 的行，避免与 watchdog 抢占、避免污染 DNS。
3. **netplan eth1 冲突**：`/etc/netplan/01-netcfg.yaml` 里 eth1 曾配了 `192.168.137.100/24`，与 usb0 同网段造成路由歧义；改为：
   ```yaml
   network:
     version: 2
     renderer: NetworkManager
     ethernets:
       eth0:
         dhcp4: true
       eth1:
         dhcp4: false
         optional: true
   ```
   改完执行 `netplan generate` 校验（务必先 `netplan generate` 再 `netplan apply`，避免把自己关在门外）。

### 4.3 一键安装（换新板 / 重装镜像后复用）

```bash
# 把 scripts/board/ 整个目录拷到板子，然后：
cd board
sudo bash install-network-fix.sh
```

`install-network-fix.sh` 会：备份现有配置 → 安装 watchdog 三件套 → 写入 connman 黑名单 → `enable + start` 服务 → 立即拉回 137.2。netplan / rc.local 属可选清理，脚本会打印提示但不强制改动（防止误操作断连）。

---

## 5. 验证

```bash
ssh ascend-dk "ip -4 addr show usb0 | grep inet; systemctl is-active dk-usb0-fix.service; uptime -p"
```

期望：`inet 192.168.137.2/24 ... usb0`、`active`。重启后 `up 0 minutes` 时即已恢复 137.2，50~70 秒后复查仍稳定 → 说明 watchdog 越过了晚到改写窗口。

---

## 6. 影响 / 副作用

- **正面**：开机 / 拔插 / 断电重来后 `192.168.137.2` 自动稳定回来，VSCode Remote-SSH 直连，无需再手动 `ip addr add`。
- **代价**：多一个几乎不耗资源的常驻进程（每 15 秒醒一次）。若你手动把 usb0 改成别的 IP，最坏 15 秒内会被 watchdog 夺回 137.2（这是设计行为）。
- **临时停用**（调试用）：`sudo systemctl stop dk-usb0-fix.service`；恢复：`sudo systemctl start dk-usb0-fix.service`。
- **彻底回滚**：`sudo systemctl disable --now dk-usb0-fix.service`，删除 `/usr/local/bin/dk-usb0-fix.sh`、`/etc/systemd/system/dk-usb0-fix.service`、`/etc/udev/rules.d/90-usb0-net.rules`，并从 `/root/netbak-20261005/` 还原 rc.local / netplan / connman main.conf。
