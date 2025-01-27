// Code generated by cudago. Edit at your own risk.
package cudago

import (
    "github.com/InternatBlackhole/cudago/cuda"
	"unsafe"
)


//here just to force usage of unsafe package
var __razdalja_ld1_useless_var__ unsafe.Pointer = nil

const (
	KeyRazdalja_ld1 = "razdalja_ld1"
)


type vectordistanceld1Args struct {
    p uintptr
    a uintptr
    b uintptr
    len int32

}

/*var (
    vectordistanceld1Args = vectordistanceld1Args{}

)*/







func VectorDistanceLD1(grid, block cuda.Dim3, p uintptr, a uintptr, b uintptr, len int32) error {
	err := autoloadLib_razdalja_ld1()
	if err != nil {
		return err
	}
	kern, err := getKernel("razdalja_ld1", "vectorDistanceLD1")
	if err != nil {
		return err
	}
	
	params := vectordistanceld1Args{
	    p: p,
	    a: a,
	    b: b,
	    len: len,
	
	}
	
	return kern.Launch(grid, block, unsafe.Pointer(&params.p), unsafe.Pointer(&params.a), unsafe.Pointer(&params.b), unsafe.Pointer(&params.len))
}

func VectorDistanceLD1Ex(grid, block cuda.Dim3, sharedMem uint64, stream *cuda.Stream, p uintptr, a uintptr, b uintptr, len int32) error {
	err := autoloadLib_razdalja_ld1()
	if err != nil {
		return err
	}
	kern, err := getKernel("razdalja_ld1", "vectorDistanceLD1")
	if err != nil {
		return err
	}
	
	params := vectordistanceld1Args{
	    p: p,
	    a: a,
	    b: b,
	    len: len,
	
	}
	
	return kern.LaunchEx(grid, block, sharedMem, stream, unsafe.Pointer(&params.p), unsafe.Pointer(&params.a), unsafe.Pointer(&params.b), unsafe.Pointer(&params.len))
}



var loaded_razdalja_ld1 = false


func autoloadLib_razdalja_ld1() error {
	if loaded_razdalja_ld1 {
		return nil
	}
	err := InitLibrary([]byte(Razdalja_ld1_ptxCode), "razdalja_ld1")
	if err != nil {
		return err
	}
	loaded_razdalja_ld1 = true
	return nil
}

const Razdalja_ld1_ptxCode = `//
// Generated by NVIDIA NVVM Compiler
//
// Compiler Build ID: CL-34431801
// Cuda compilation tools, release 12.6, V12.6.20
// Based on NVVM 7.0.1
//

.version 8.5
.target sm_52
.address_size 64

	// .globl	vectorDistanceLD1
.extern .shared .align 16 .b8 part[];

.visible .entry vectorDistanceLD1(
	.param .u64 vectorDistanceLD1_param_0,
	.param .u64 vectorDistanceLD1_param_1,
	.param .u64 vectorDistanceLD1_param_2,
	.param .u32 vectorDistanceLD1_param_3
)
{
	.reg .pred 	%p<9>;
	.reg .f32 	%f<36>;
	.reg .b32 	%r<28>;
	.reg .b64 	%rd<25>;


	ld.param.u64 	%rd9, [vectorDistanceLD1_param_0];
	ld.param.u64 	%rd10, [vectorDistanceLD1_param_1];
	ld.param.u64 	%rd11, [vectorDistanceLD1_param_2];
	ld.param.u32 	%r17, [vectorDistanceLD1_param_3];
	mov.u32 	%r1, %tid.x;
	mul.wide.u32 	%rd12, %r1, 4;
	mov.u64 	%rd13, part;
	add.s64 	%rd1, %rd13, %rd12;
	mov.u32 	%r18, 0;
	st.shared.u32 	[%rd1], %r18;
	mov.u32 	%r2, %ntid.x;
	mov.u32 	%r3, %ctaid.x;
	mad.lo.s32 	%r23, %r3, %r2, %r1;
	setp.ge.s32 	%p1, %r23, %r17;
	@%p1 bra 	$L__BB0_4;

	mov.u32 	%r19, %nctaid.x;
	mul.lo.s32 	%r5, %r19, %r2;
	cvta.to.global.u64 	%rd2, %rd10;
	cvta.to.global.u64 	%rd3, %rd11;
	mov.f32 	%f30, 0f00000000;

$L__BB0_2:
	mul.wide.s32 	%rd14, %r23, 4;
	add.s64 	%rd15, %rd2, %rd14;
	add.s64 	%rd16, %rd3, %rd14;
	ld.global.nc.f32 	%f11, [%rd16];
	ld.global.nc.f32 	%f12, [%rd15];
	sub.f32 	%f13, %f12, %f11;
	fma.rn.f32 	%f30, %f13, %f13, %f30;
	add.s32 	%r23, %r23, %r5;
	setp.lt.s32 	%p2, %r23, %r17;
	@%p2 bra 	$L__BB0_2;

	st.shared.f32 	[%rd1], %f30;

$L__BB0_4:
	bar.sync 	0;
	setp.ne.s32 	%p3, %r1, 0;
	@%p3 bra 	$L__BB0_13;

	setp.eq.s32 	%p4, %r2, 0;
	mov.f32 	%f35, 0f00000000;
	@%p4 bra 	$L__BB0_12;

	add.s32 	%r21, %r2, -1;
	and.b32  	%r27, %r2, 3;
	setp.lt.u32 	%p5, %r21, 3;
	mov.f32 	%f35, 0f00000000;
	mov.u32 	%r26, 0;
	@%p5 bra 	$L__BB0_9;

	sub.s32 	%r25, %r2, %r27;
	mov.f32 	%f35, 0f00000000;
	mov.u32 	%r26, 0;
	mov.u64 	%rd23, %rd13;

$L__BB0_8:
	ld.shared.v4.f32 	{%f18, %f19, %f20, %f21}, [%rd23];
	add.f32 	%f26, %f35, %f18;
	add.f32 	%f27, %f26, %f19;
	add.f32 	%f28, %f27, %f20;
	add.f32 	%f35, %f28, %f21;
	add.s32 	%r26, %r26, 4;
	add.s64 	%rd23, %rd23, 16;
	add.s32 	%r25, %r25, -4;
	setp.ne.s32 	%p6, %r25, 0;
	@%p6 bra 	$L__BB0_8;

$L__BB0_9:
	setp.eq.s32 	%p7, %r27, 0;
	@%p7 bra 	$L__BB0_12;

	mul.wide.s32 	%rd18, %r26, 4;
	add.s64 	%rd24, %rd13, %rd18;

$L__BB0_11:
	.pragma "nounroll";
	ld.shared.f32 	%f29, [%rd24];
	add.f32 	%f35, %f35, %f29;
	add.s64 	%rd24, %rd24, 4;
	add.s32 	%r27, %r27, -1;
	setp.ne.s32 	%p8, %r27, 0;
	@%p8 bra 	$L__BB0_11;

$L__BB0_12:
	cvta.to.global.u64 	%rd20, %rd9;
	mul.wide.u32 	%rd21, %r3, 4;
	add.s64 	%rd22, %rd20, %rd21;
	st.global.f32 	[%rd22], %f35;

$L__BB0_13:
	ret;

}

`
