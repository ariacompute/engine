#!/bin/sh
# check-emudhra.sh — 离线检测 CA 信任库是否仍信任受 eMudhra/emSign 私钥泄露影响的证书。
# 不依赖网络/代理，直接按 SHA-256 指纹比对。
#
# 方法论（重要）：
#   信任库里存放的是“根证书”（信任锚），而不是中间证书。一张被伪造的站点证书只要能
#   链到库里仍存在的 emSign 根，就会被信任。因此本脚本以【emSign 根】为决定性判定：
#   库中存在任一 emSign 根 => 仍未修补（FAIL）。
#   8 张弱私钥中间证书作为次级提示一并检查（干净库不应装运它们）。
#   注意：在线探测（如 curl https://dv-g3b-emudhra.we-have-a-problem.com）在前向/拦截
#   代理环境下会返回 200 假阳性，请以本离线脚本为准。
#
# 用法:
#   scripts/check-emudhra.sh                 # 扫描系统默认 bundle
#   scripts/check-emudhra.sh /path/ca.crt ... # 扫描指定文件（如容器内 /etc/ssl/certs/ca-certificates.crt）
#
# 退出码: 0 = 未发现受影响证书；1 = 发现仍被信任的受影响证书；2 = 用法/环境错误。
#
# 另注：本仓库 Rust 二进制（aria-engine/aria-cli）的 TLS 信任库来自编译进二进制的
# webpki-roots（>=1.0.10），不走系统 bundle；请另用
#   cargo tree -i webpki-roots   # 须 >= 1.0.10
# 核验。语言绑定（Go/Python/TS/RN）仅做 FFI，无出向 TLS，不在本脚本范围。

set -u

# emSign 根证书（信任锚） SHA-256 指纹。来源: eMudhra 事件公开公告 / CCADB。
ROOTS="7DD78D5F4F13459A83DFF9ABBBA62EDBAF6F2D102BF257FD712F4D9F2746ED8D
86A1ECBA089C4A8D3BBE2734C612BA341D813E043CF9E8A862CD5C57A36BBE6B"

# 8 张受影响 emSign 中间证书 SHA-256 指纹（私钥可反推）。
# 来源: Kudelski Security advisory on eMudhra Intermediate CA Private Key Compromise.
INTERMEDIATES="42C4ACCD12C4F59597925E1A08A7A90102608543809308D50EC427E118BA1493
0164D7CEAE5BB909C272AF151FF44B1177AE9B47D9424B94F1D9558C30934D19
E7568FE6E12331F4FD7DE61DA4C7559D0F8EB192A218D4741A3BA75184333838
432EC25F0D28DE9006D97BF752F17BBB2CD41F669933FCB072D9A1148D0261F3
98B896625C0C230ADEEC3EB2CF6AAD5001ECF9156A68644E82374CE5833BC25B
CB6E15165AC802E9B3F993F16C25D41F4E81C21EBDE71AE1B012AC00BF9C7C3F
9E39DE2881273ACC225A149C72BA3E0F51424963669AB633EE1DC7DF5DB067B4
4B2460A466F95FD40113BCDCCA2CEB29B3B093921DE45474DA7DF48874C630B1"

DEFAULT_FILES="/etc/ssl/certs/ca-certificates.crt
/etc/pki/tls/certs/ca-bundle.crt
/etc/ssl/cert.pem"

FILES="${*:-$DEFAULT_FILES}"

if ! command -v openssl >/dev/null 2>&1; then
  echo "error: openssl 未安装，无法扫描 bundle" >&2
  exit 2
fi

# 提取给定文件里所有证书的 SHA-256 指纹（大写，无冒号）。
fingerprints_of() {
  openssl crl2pkcs7 -nocrl -certfile "$1" 2>/dev/null \
    | openssl pkcs7 -print_certs -noout 2>/dev/null \
    | tr 'a-f' 'A-F'
}

hit=0
for f in $FILES; do
  [ -f "$f" ] || continue
  fps=$(fingerprints_of "$f")
  for fp in $ROOTS; do
    if echo "$fps" | grep -q "$fp"; then
      echo "FAIL(ROOT): 信任库仍包含 emSign 根证书: $fp  (文件: $f)"
      hit=1
    fi
  done
  for fp in $INTERMEDIATES; do
    if echo "$fps" | grep -q "$fp"; then
      echo "WARN(INTERMEDIATE): 信任库仍装运受影响中间证书: $fp  (文件: $f)"
      hit=1
    fi
  done
done

if [ "$hit" -eq 0 ]; then
  echo "OK: 在指定信任库中未发现 emSign 根或受影响中间证书"
  exit 0
fi
exit 1
