#!/usr/bin/env bash
# Sets local CPU/platform power knobs for repeatable gemma.cpp benchmarks.

set -euo pipefail

# Accept SUDO_PASSWORD for repeatable local benchmark setup. The script still
# works in a normal interactive shell when the variable is not set.
run_sudo() {
  if [[ -n "${SUDO_PASSWORD:-}" ]]; then
    printf '%s\n' "${SUDO_PASSWORD}" | sudo -S "$@"
  else
    sudo "$@"
  fi
}

if command -v powerprofilesctl >/dev/null 2>&1; then
  powerprofilesctl set performance || true
fi

# Platform profile and cpufreq/EPP are separate knobs on this laptop. Set all of
# them because leaving any one in a low-power mode was enough to skew PaliGemma2
# timings during baseline collection.
if [[ -w /sys/firmware/acpi/platform_profile ]]; then
  echo performance > /sys/firmware/acpi/platform_profile
elif [[ -e /sys/firmware/acpi/platform_profile ]]; then
  run_sudo bash -c 'echo performance > /sys/firmware/acpi/platform_profile'
fi

CPUPOWER="${CPUPOWER:-}"
if [[ -z "${CPUPOWER}" ]]; then
  if [[ -x /usr/lib/linux-oem-6.17-tools-6.17.0-1020/cpupower ]]; then
    CPUPOWER=/usr/lib/linux-oem-6.17-tools-6.17.0-1020/cpupower
  elif command -v cpupower >/dev/null 2>&1; then
    CPUPOWER=cpupower
  fi
fi

if [[ -n "${CPUPOWER}" ]]; then
  run_sudo "${CPUPOWER}" frequency-set -g performance
fi

run_sudo bash -c 'for f in /sys/devices/system/cpu/cpufreq/policy*/energy_performance_preference; do [ -w "$f" ] && echo performance > "$f"; done'

# Print a compact confirmation for the representative CPU policies observed on
# this heterogeneous system: fast-core and dense-core clusters plus SMT siblings.
echo "Power profile: $(cat /sys/firmware/acpi/platform_profile 2>/dev/null || echo unknown)"
for p in /sys/devices/system/cpu/cpufreq/policy{0,4,12,16}; do
  [[ -d "${p}" ]] || continue
  printf '%s governor=' "${p##*/}"
  cat "${p}/scaling_governor"
  if [[ -r "${p}/energy_performance_preference" ]]; then
    printf '%s epp=' "${p##*/}"
    cat "${p}/energy_performance_preference"
  fi
done
