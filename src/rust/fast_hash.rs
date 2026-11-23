//! Small multiplicative hasher for maps keyed by small integers where
//! iteration order cannot influence generated code or results.

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};

#[derive(Default)]
pub struct FastHasher(u64);

impl FastHasher {
    #[inline]
    fn mix(&mut self, x: u64) { self.0 = (self.0 ^ x).wrapping_mul(0x9E37_79B9_7F4A_7C15); }
}

impl Hasher for FastHasher {
    #[inline]
    fn finish(&self) -> u64 {
        let h = self.0;
        h ^ (h >> 29)
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.mix(b as u64);
        }
    }

    #[inline]
    fn write_u8(&mut self, x: u8) { self.mix(x as u64); }
    #[inline]
    fn write_u16(&mut self, x: u16) { self.mix(x as u64); }
    #[inline]
    fn write_u32(&mut self, x: u32) { self.mix(x as u64); }
    #[inline]
    fn write_u64(&mut self, x: u64) { self.mix(x); }
    #[inline]
    fn write_usize(&mut self, x: usize) { self.mix(x as u64); }
}

pub type FastHashMap<K, V> = HashMap<K, V, BuildHasherDefault<FastHasher>>;
#[allow(dead_code)]
pub type FastHashSet<K> = HashSet<K, BuildHasherDefault<FastHasher>>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_keys_spread_over_low_bits() {
        for bits in [8u32, 6, 4] {
            let mask = (1u64 << bits) - 1;
            let mut seen = HashSet::new();
            for key in 0..(1u64 << bits) * 4 {
                let mut h = FastHasher::default();
                h.write_u32(key as u32);
                seen.insert(h.finish() & mask);
            }
            assert!(seen.len() as u64 * 4 >= (1 << bits) * 3);
        }
    }

    #[test]
    fn small_keys_spread_over_bits_25_to_31() {
        let mut seen = HashSet::new();
        for key in 0..4096u32 {
            let mut h = FastHasher::default();
            h.write_u32(key);
            seen.insert((h.finish() >> 25) & 0x7f);
        }
        assert!(seen.len() >= 96);
    }
}
