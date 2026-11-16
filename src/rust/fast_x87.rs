//! Explicit performance policy. Architectural storage stays raw80, while
//! ordinary masked, nearest-rounded arithmetic uses native Wasm binary64.
//! Precision-control bits do not constrain these approximate intermediates.
//! Exceptional encodings/ranges retain the existing SoftFloat path.
use crate::cpu::global_pointers::fpu_control_word;
use crate::softfloat::F80;

static mut MODE: u32 = 0;

/// 0: existing arithmetic; 1: binary64 helpers; 2: helpers plus JIT value runs;
/// 3: same as 2 with a diagnostic instruction counter (off in production).
/// The host changes policy only between execution slices. Clear compiled code
/// before selecting a policy whose JIT organization may differ.
#[no_mangle]
pub unsafe fn vine_set_x87_mode(mode: u32) -> u32 {
    if mode > 3 || crate::cpu::cpu::vine_execution_is_active() {
        return 1;
    }
    if MODE != mode {
        crate::jit::jit_clear_cache_js();
        MODE = mode;
    }
    0
}
#[no_mangle]
pub unsafe fn vine_get_x87_mode() -> u32 { MODE }

#[inline(always)]
pub fn enabled() -> bool { unsafe { MODE != 0 && *fpu_control_word & 0x0c3f == 0x003f } }

/// No SoftFloat calls and no change to architectural or exception state.
#[inline(always)]
pub fn to_native(value: F80) -> Option<f64> {
    let exponent = value.sign_exponent & 0x7fff;
    let sign = u64::from(value.sign_exponent & 0x8000) << 48;
    if exponent == 0 && value.mantissa == 0 {
        return Some(f64::from_bits(sign));
    }
    if !(0x3c01..=0x43fe).contains(&exponent) || value.mantissa >> 63 == 0 {
        return None;
    }
    let mut significand = value.mantissa >> 11;
    let tail = value.mantissa & 2047;
    significand += u64::from(tail > 1024 || tail == 1024 && significand & 1 != 0);
    let mut exponent = u64::from(exponent - 0x3c00);
    if significand == 1 << 53 {
        significand >>= 1;
        exponent += 1;
    }
    if exponent == 2047 {
        return None;
    }
    Some(f64::from_bits(
        sign | exponent << 52 | significand & ((1 << 52) - 1),
    ))
}

#[inline(always)]
pub fn from_native(value: f64) -> Option<F80> {
    let bits = value.to_bits();
    let exponent = (bits >> 52 & 0x7ff) as u16;
    let sign = (bits >> 48) as u16 & 0x8000;
    if bits << 1 == 0 {
        return Some(F80 {
            mantissa: 0,
            sign_exponent: sign,
        });
    }
    if exponent == 0 || exponent == 2047 {
        return None;
    }
    Some(F80 {
        mantissa: (bits << 11) | 1 << 63,
        sign_exponent: sign | (exponent + 0x3c00),
    })
}

#[inline(always)]
pub fn binary(a: F80, b: F80, operation: u8) -> Option<F80> {
    if !enabled() {
        return None;
    }
    let x = to_native(a)?;
    let y = to_native(b)?;
    if operation == 3 && y == 0.0 {
        return None;
    }
    let result = match operation {
        0 => x + y,
        1 => x - y,
        2 => x * y,
        3 => x / y,
        _ => unreachable!(),
    };
    if result == 0.0 && operation >= 2 && x != 0.0 && y != 0.0 {
        return None;
    }
    from_native(result)
}

#[inline(always)]
pub fn sqrt(a: F80) -> Option<F80> {
    if !enabled() {
        return None;
    }
    let x = to_native(a)?;
    if x < 0.0 {
        return None;
    }
    from_native(x.sqrt())
}

/// Integer stores retain range checks; never replace the indefinite integer
/// result for invalid conversions with Rust's saturating float cast.
#[inline(always)]
pub fn integer(a: F80, truncate: bool) -> Option<i64> {
    if !enabled() {
        return None;
    }
    let x = to_native(a)?;
    let rounded = if truncate { x.trunc() } else { x.round_ties_even() };
    if !(-9223372036854775808.0..9223372036854775808.0).contains(&rounded) {
        return None;
    }
    Some(rounded as i64)
}
