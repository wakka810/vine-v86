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
  (import "e" "exit_jit" (func $e.exit_jit (type $t0)))
  (import "e" "m" (memory {normalised output}))
  (func $f (export "f") (type $t1) (param $p0 i32)
    (local $l1 i32) (local $l2 i32) (local $l3 i32) (local $l4 i32) (local $l5 i32) (local $l6 i32) (local $l7 i32) (local $l8 i32) (local $l9 i32) (local $l10 i32)
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
                  (i32.const 1))
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
                (i32.const 1)))
            (i32.store
              (i32.const 556)
              (i32.add
                (i32.or
                  (i32.and
                    (i32.load
                      (i32.const 556))
                    (i32.const -4096))
                  (i32.const 5))
                (i32.const 4074)))
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
