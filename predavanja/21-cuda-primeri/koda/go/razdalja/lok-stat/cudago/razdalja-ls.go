// Code generated by cudago. Edit at your own risk.
package cudago

import (
    "github.com/InternatBlackhole/cudago/cuda"
	"unsafe"
)


//here just to force usage of unsafe package
var __razdalja_ls_useless_var__ unsafe.Pointer = nil

const (
	KeyRazdalja_ls = "razdalja_ls"
)


type vectordistancelsArgs struct {
    p uintptr
    a uintptr
    b uintptr
    len int32

}

/*var (
    vectordistancelsArgs = vectordistancelsArgs{}

)*/







func VectorDistanceLS(grid, block cuda.Dim3, p uintptr, a uintptr, b uintptr, len int32) error {
	err := autoloadLib_razdalja_ls()
	if err != nil {
		return err
	}
	kern, err := getKernel("razdalja_ls", "vectorDistanceLS")
	if err != nil {
		return err
	}
	
	params := vectordistancelsArgs{
	    p: p,
	    a: a,
	    b: b,
	    len: len,
	
	}
	
	return kern.Launch(grid, block, unsafe.Pointer(&params.p), unsafe.Pointer(&params.a), unsafe.Pointer(&params.b), unsafe.Pointer(&params.len))
}

func VectorDistanceLSEx(grid, block cuda.Dim3, sharedMem uint64, stream *cuda.Stream, p uintptr, a uintptr, b uintptr, len int32) error {
	err := autoloadLib_razdalja_ls()
	if err != nil {
		return err
	}
	kern, err := getKernel("razdalja_ls", "vectorDistanceLS")
	if err != nil {
		return err
	}
	
	params := vectordistancelsArgs{
	    p: p,
	    a: a,
	    b: b,
	    len: len,
	
	}
	
	return kern.LaunchEx(grid, block, sharedMem, stream, unsafe.Pointer(&params.p), unsafe.Pointer(&params.a), unsafe.Pointer(&params.b), unsafe.Pointer(&params.len))
}



var loaded_razdalja_ls = false


func autoloadLib_razdalja_ls() error {
	if loaded_razdalja_ls {
		return nil
	}
	err := InitLibrary([]byte(Razdalja_ls_ptxCode), "razdalja_ls")
	if err != nil {
		return err
	}
	loaded_razdalja_ls = true
	return nil
}

const Razdalja_ls_ptxCode = `//
// Generated by NVIDIA NVVM Compiler
//
// Compiler Build ID: CL-34431801
// Cuda compilation tools, release 12.6, V12.6.20
// Based on NVVM 7.0.1
//

.version 8.5
.target sm_52
.address_size 64

	// .globl	vectorDistanceLS
// _ZZ16vectorDistanceLSE4part has been demoted

.visible .entry vectorDistanceLS(
	.param .u64 vectorDistanceLS_param_0,
	.param .u64 vectorDistanceLS_param_1,
	.param .u64 vectorDistanceLS_param_2,
	.param .u32 vectorDistanceLS_param_3
)
{
	.reg .pred 	%p<9>;
	.reg .f32 	%f<31>;
	.reg .b32 	%r<29>;
	.reg .b64 	%rd<25>;
	// demoted variable
	.shared .align 4 .b8 _ZZ16vectorDistanceLSE4part[4096];

	ld.param.u64 	%rd10, [vectorDistanceLS_param_0];
	ld.param.u64 	%rd11, [vectorDistanceLS_param_1];
	ld.param.u64 	%rd12, [vectorDistanceLS_param_2];
	ld.param.u32 	%r17, [vectorDistanceLS_param_3];
	mov.u32 	%r1, %tid.x;
	mul.wide.u32 	%rd13, %r1, 4;
	mov.u64 	%rd14, _ZZ16vectorDistanceLSE4part;
	add.s64 	%rd1, %rd14, %rd13;
	mov.u32 	%r18, 0;
	st.shared.u32 	[%rd1], %r18;
	mov.u32 	%r2, %ntid.x;
	mov.u32 	%r3, %ctaid.x;
	mad.lo.s32 	%r24, %r3, %r2, %r1;
	setp.ge.s32 	%p1, %r24, %r17;
	@%p1 bra 	$L__BB0_4;

	mov.u32 	%r19, %nctaid.x;
	mul.lo.s32 	%r5, %r19, %r2;
	cvta.to.global.u64 	%rd2, %rd11;
	cvta.to.global.u64 	%rd3, %rd12;
	mov.f32 	%f25, 0f00000000;

$L__BB0_2:
	mul.wide.s32 	%rd15, %r24, 4;
	add.s64 	%rd16, %rd2, %rd15;
	add.s64 	%rd17, %rd3, %rd15;
	ld.global.nc.f32 	%f11, [%rd17];
	ld.global.nc.f32 	%f12, [%rd16];
	sub.f32 	%f13, %f12, %f11;
	fma.rn.f32 	%f25, %f13, %f13, %f25;
	add.s32 	%r24, %r24, %r5;
	setp.lt.s32 	%p2, %r24, %r17;
	@%p2 bra 	$L__BB0_2;

	st.shared.f32 	[%rd1], %f25;

$L__BB0_4:
	bar.sync 	0;
	setp.ne.s32 	%p3, %r1, 0;
	@%p3 bra 	$L__BB0_13;

	cvta.to.global.u64 	%rd18, %rd10;
	mul.wide.u32 	%rd19, %r3, 4;
	add.s64 	%rd4, %rd18, %rd19;
	mov.u32 	%r27, 0;
	st.global.u32 	[%rd4], %r27;
	setp.eq.s32 	%p4, %r2, 0;
	@%p4 bra 	$L__BB0_13;

	add.s32 	%r22, %r2, -1;
	and.b32  	%r28, %r2, 3;
	setp.lt.u32 	%p5, %r22, 3;
	mov.f32 	%f30, 0f00000000;
	@%p5 bra 	$L__BB0_9;

	sub.s32 	%r26, %r2, %r28;
	mov.u32 	%r27, 0;
	mov.f32 	%f30, 0f00000000;
	mov.u64 	%rd23, %rd14;

$L__BB0_8:
	ld.shared.f32 	%f17, [%rd23];
	add.f32 	%f18, %f17, %f30;
	ld.shared.f32 	%f19, [%rd23+4];
	add.f32 	%f20, %f19, %f18;
	ld.shared.f32 	%f21, [%rd23+8];
	add.f32 	%f22, %f21, %f20;
	ld.shared.f32 	%f23, [%rd23+12];
	add.f32 	%f30, %f23, %f22;
	add.s32 	%r27, %r27, 4;
	add.s64 	%rd23, %rd23, 16;
	add.s32 	%r26, %r26, -4;
	setp.ne.s32 	%p6, %r26, 0;
	@%p6 bra 	$L__BB0_8;

$L__BB0_9:
	setp.eq.s32 	%p7, %r28, 0;
	@%p7 bra 	$L__BB0_12;

	mul.wide.s32 	%rd21, %r27, 4;
	add.s64 	%rd24, %rd14, %rd21;

$L__BB0_11:
	.pragma "nounroll";
	ld.shared.f32 	%f24, [%rd24];
	add.f32 	%f30, %f24, %f30;
	add.s64 	%rd24, %rd24, 4;
	add.s32 	%r28, %r28, -1;
	setp.ne.s32 	%p8, %r28, 0;
	@%p8 bra 	$L__BB0_11;

$L__BB0_12:
	st.global.f32 	[%rd4], %f30;

$L__BB0_13:
	ret;

}

`
