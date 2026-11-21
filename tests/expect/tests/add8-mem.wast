(module
  (type $t0 (func))
  (type $t1 (func (param i32)))
  (type $t2 (func (param i32 i32)))
  (type $t3 (func (param i32 i32 i32)))
  (type $t4 (func (result i32)))
  (type $t5 (func (result i64)))
  (type $t6 (func (param i32) (result i32)))
  (type $t7 (func (param i32 i32) (result i32)))
  (type $t8 (func (param f32) (result i32)))
  (type $t9 (func (param f64) (result i32)))
  (type $t10 (func (param i32 i64)))
  (type $t11 (func (param i64 i32)))
  (type $t12 (func (param i64 i32) (result i32)))
  (type $t13 (func (param i64 i32) (result i64)))
  (type $t14 (func (param f32 i32)))
  (type $t15 (func (param i32 i32 i32) (result i32)))
  (type $t16 (func (param i64 i32 i32)))
  (type $t17 (func (param i32 i64 i32)))
  (type $t18 (func (param i32 i64 i32) (result i32)))
  (type $t19 (func (param i32 i64 i64 i32) (result i32)))
  (type $t20 (func (param i64 i32 i64 i32) (result i64)))
  (type $t21 (func (param i64 i32 i64 i32)))
  (type $t22 (func (param i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)))
  (import "e" "vine_jit_instruction_limit" (func $e.vine_jit_instruction_limit (type $t4)))
  (import "e" "trigger_gp_jit" (func $e.trigger_gp_jit (type $t2)))
  (import "e" "safe_read_write8_slow_jit" (func $e.safe_read_write8_slow_jit (type $t7)))
  (import "e" "safe_write8_slow_jit" (func $e.safe_write8_slow_jit (type $t15)))
  (import "e" "bug_gen_safe_read_write_page_fault" (func $e.bug_gen_safe_read_write_page_fault (type $t2)))
  (import "e" "instr_F4" (func $e.instr_F4 (type $t0)))
  (import "e" "exit_jit" (func $e.exit_jit (type $t0)))
  (import "e" "m" (memory {normalised output}))
  (func $f (export "f") (type $t1) (param $p0 i32)
    (local $l1 i32) (local $l2 i32) (local $l3 i32) (local $l4 i32) (local $l5 i32) (local $l6 i32) (local $l7 i32) (local $l8 i32) (local $l9 i32) (local $l10 i32) (local $l11 i32) (local $l12 i32) (local $l13 i32) (local $l14 i32)
    (local.set $l1
      (i32.load
        (i32.const 64)))
    (local.set $l2
      (i32.load
        (i32.const 68)))
    (local.set $l3
      (i32.load
        (i32.const 72)))
    (local.set $l4
      (i32.load
        (i32.const 76)))
    (local.set $l5
      (i32.load
        (i32.const 80)))
    (local.set $l6
      (i32.load
        (i32.const 84)))
    (local.set $l7
      (i32.load
        (i32.const 88)))
    (local.set $l8
      (i32.load
        (i32.const 92)))
    (local.set $l9
      (i32.const 0))
    (local.set $l10
      (call $e.vine_jit_instruction_limit))
    (local.set $l10
      (select
        (local.get $l10)
        (i32.const 100003)
        (i32.lt_u
          (local.get $l10)
          (i32.const 100003))))
    (block $B0
      (block $B1
        (loop $L2
          (block $B3
            (block $B4
            )
            (if $I5
              (i32.gt_u
                (i32.add
                  (local.get $l9)
                  (i32.const 2))
                (local.get $l10))
              (then
                (i32.store
                  (i32.const 556)
                  (i32.or
                    (i32.and
                      (i32.load
                        (i32.const 556))
                      (i32.const -4096))
                    (i32.const 0)))
                (br $B0)))
            (local.set $l9
              (i32.add
                (local.get $l9)
                (i32.const 2)))
            (local.get $l1)
            (if $I6
              (i32.load8_u
                (i32.const 727))
              (then
                (call $e.trigger_gp_jit
                  (i32.const 0)
                  (i32.const 0))
                (br $B1)))
            (i32.load
              (i32.const 748))
            (local.set $l11
              (i32.add))
            (block $B7
              (br_if $B7
                (local.tee $l13
                  (i32.eq
                    (i32.and
                      (local.tee $l12
                        (i32.load offset={normalised output}
                          (i32.shl
                            (i32.shr_u
                              (local.get $l11)
                              (i32.const 12))
                            (i32.const 2))))
                      (i32.const 4075))
                    (i32.const 1))))
              (if $I8
                (i32.and
                  (local.tee $l12
                    (call $e.safe_read_write8_slow_jit
                      (local.get $l11)
                      (i32.const 58916864)))
                  (i32.const 1))
                (then
                  (local.set $l9
                    (i32.sub
                      (local.get $l9)
                      (i32.mul
                        (i32.const 1)
                        (i32.and
                          (i32.shr_u
                            (local.get $l12)
                            (i32.const 1))
                          (i32.const 1)))))
                  (br $B1))))
            (local.set $l14
              (i32.load8_u
                (local.tee $l12
                  (i32.xor
                    (i32.and
                      (local.get $l12)
                      (i32.const -4096))
                    (local.get $l11)))))
            (i32.store
              (i32.const 104)
              (i32.and
                (local.get $l14)
                (i32.const 255)))
            (i32.store
              (i32.const 112)
              (i32.and
                (i32.add
                  (local.get $l14)
                  (local.get $l3))
                (i32.const 255)))
            (i64.store
              (i32.const 96)
              (i64.const 9710921056263))
            (local.set $l14
              (i32.load8_u
                (i32.const 112)))
            (if $I9
              (i32.eqz
                (local.get $l13))
              (then
                (if $I10
                  (i32.and
                    (call $e.safe_write8_slow_jit
                      (local.get $l11)
                      (local.get $l14)
                      (i32.const 58916864))
                    (i32.const 1))
                  (then
                    (call $e.bug_gen_safe_read_write_page_fault
                      (i32.const 8)
                      (local.get $l11))))))
            (i32.store8
              (local.get $l12)
              (local.get $l14))
            (i32.store
              (i32.const 560)
              (i32.or
                (i32.and
                  (i32.load
                    (i32.const 556))
                  (i32.const -4096))
                (i32.const 2)))
            (i32.store
              (i32.const 556)
              (i32.or
                (i32.and
                  (i32.load
                    (i32.const 556))
                  (i32.const -4096))
                (i32.const 3)))
            (i32.store
              (i32.const 64)
              (local.get $l1))
            (i32.store
              (i32.const 68)
              (local.get $l2))
            (i32.store
              (i32.const 72)
              (local.get $l3))
            (i32.store
              (i32.const 76)
              (local.get $l4))
            (i32.store
              (i32.const 80)
              (local.get $l5))
            (i32.store
              (i32.const 84)
              (local.get $l6))
            (i32.store
              (i32.const 88)
              (local.get $l7))
            (i32.store
              (i32.const 92)
              (local.get $l8))
            (call $e.instr_F4)
            (local.set $l1
              (i32.load
                (i32.const 64)))
            (local.set $l2
              (i32.load
                (i32.const 68)))
            (local.set $l3
              (i32.load
                (i32.const 72)))
            (local.set $l4
              (i32.load
                (i32.const 76)))
            (local.set $l5
              (i32.load
                (i32.const 80)))
            (local.set $l6
              (i32.load
                (i32.const 84)))
            (local.set $l7
              (i32.load
                (i32.const 88)))
            (local.set $l8
              (i32.load
                (i32.const 92)))
            (br $B0))
          (unreachable)))
      (i32.store
        (i32.const 64)
        (local.get $l1))
      (i32.store
        (i32.const 68)
        (local.get $l2))
      (i32.store
        (i32.const 72)
        (local.get $l3))
      (i32.store
        (i32.const 76)
        (local.get $l4))
      (i32.store
        (i32.const 80)
        (local.get $l5))
      (i32.store
        (i32.const 84)
        (local.get $l6))
      (i32.store
        (i32.const 88)
        (local.get $l7))
      (i32.store
        (i32.const 92)
        (local.get $l8))
      (call $e.exit_jit)
      (i32.store
        (i32.const 664)
        (i32.add
          (i32.load
            (i32.const 664))
          (local.get $l9)))
      (return))
    (i32.store
      (i32.const 64)
      (local.get $l1))
    (i32.store
      (i32.const 68)
      (local.get $l2))
    (i32.store
      (i32.const 72)
      (local.get $l3))
    (i32.store
      (i32.const 76)
      (local.get $l4))
    (i32.store
      (i32.const 80)
      (local.get $l5))
    (i32.store
      (i32.const 84)
      (local.get $l6))
    (i32.store
      (i32.const 88)
      (local.get $l7))
    (i32.store
      (i32.const 92)
      (local.get $l8))
    (i32.store
      (i32.const 664)
      (i32.add
        (i32.load
          (i32.const 664))
        (local.get $l9)))))
