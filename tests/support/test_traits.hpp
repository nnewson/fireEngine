#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace test_traits
{

template <typename T, typename... Args>
inline constexpr bool nothrow_constructible_from_v = std::is_nothrow_constructible_v<T, Args...>;

template <typename T, typename U>
inline constexpr bool nothrow_assignable_from_v = std::is_nothrow_assignable_v<T, U>;

template <typename T>
concept has_nothrow_equality = requires(const T& lhs, const T& rhs) {
    { lhs == rhs } noexcept -> std::same_as<bool>;
};

template <typename T>
concept has_nothrow_vec2_accessors = requires(T value, const T& const_value) {
    { const_value.s() } noexcept -> std::same_as<float>;
    { const_value.t() } noexcept -> std::same_as<float>;
    { value.s(1.0f) } noexcept -> std::same_as<void>;
    { value.t(1.0f) } noexcept -> std::same_as<void>;
};

template <typename T>
concept has_nothrow_vec3_accessors = requires(T value, const T& const_value) {
    { const_value.x() } noexcept -> std::same_as<float>;
    { const_value.y() } noexcept -> std::same_as<float>;
    { const_value.z() } noexcept -> std::same_as<float>;
    { value.x(1.0f) } noexcept -> std::same_as<void>;
    { value.y(1.0f) } noexcept -> std::same_as<void>;
    { value.z(1.0f) } noexcept -> std::same_as<void>;
};

template <typename T>
concept has_nothrow_vec4_accessors = requires(T value, const T& const_value) {
    { const_value.x() } noexcept -> std::same_as<float>;
    { const_value.y() } noexcept -> std::same_as<float>;
    { const_value.z() } noexcept -> std::same_as<float>;
    { const_value.w() } noexcept -> std::same_as<float>;
    { value.x(1.0f) } noexcept -> std::same_as<void>;
    { value.y(1.0f) } noexcept -> std::same_as<void>;
    { value.z(1.0f) } noexcept -> std::same_as<void>;
    { value.w(1.0f) } noexcept -> std::same_as<void>;
};

template <typename T>
concept has_nothrow_vec_arithmetic = requires(T lhs, const T& rhs) {
    { lhs + rhs } noexcept -> std::same_as<T>;
    { lhs - rhs } noexcept -> std::same_as<T>;
    { lhs * 1.0f } noexcept -> std::same_as<T>;
    { lhs / 1.0f } noexcept -> std::same_as<T>;
    { lhs += rhs } noexcept -> std::same_as<T&>;
    { lhs -= rhs } noexcept -> std::same_as<T&>;
    { lhs *= 1.0f } noexcept -> std::same_as<T&>;
    { lhs /= 1.0f } noexcept -> std::same_as<T&>;
};

template <typename T>
concept has_nothrow_vec_common_math = requires(T value, const T& const_value) {
    { T::dotProduct(const_value, const_value) } noexcept -> std::same_as<float>;
    { const_value.dotProduct(const_value) } noexcept -> std::same_as<float>;
    { const_value.magnitude() } noexcept -> std::same_as<float>;
    { const_value.magnitudeSquared() } noexcept -> std::same_as<float>;
    { T::normalise(const_value) } noexcept -> std::same_as<T>;
    { value.normalise() } noexcept -> std::same_as<T&>;
};

template <typename T>
concept has_nothrow_vec3_cross_product = requires(const T& lhs, const T& rhs) {
    { T::crossProduct(lhs, rhs) } noexcept -> std::same_as<T>;
    { lhs.crossProduct(rhs) } noexcept -> std::same_as<T>;
};

template <typename T>
concept has_nothrow_colour3_accessors = requires(T value, const T& const_value) {
    { const_value.r() } noexcept -> std::same_as<float>;
    { const_value.g() } noexcept -> std::same_as<float>;
    { const_value.b() } noexcept -> std::same_as<float>;
    { value.r(0.1f) } noexcept -> std::same_as<void>;
    { value.g(0.1f) } noexcept -> std::same_as<void>;
    { value.b(0.1f) } noexcept -> std::same_as<void>;
};

template <typename T>
concept has_nothrow_joints4_getters = requires(const T& const_value) {
    { const_value.j0() } noexcept -> std::same_as<uint32_t>;
    { const_value.j1() } noexcept -> std::same_as<uint32_t>;
    { const_value.j2() } noexcept -> std::same_as<uint32_t>;
    { const_value.j3() } noexcept -> std::same_as<uint32_t>;
};

template <typename T>
concept has_nothrow_joints4_setters = requires(T value) {
    { value.j0(0U) } noexcept -> std::same_as<void>;
    { value.j1(0U) } noexcept -> std::same_as<void>;
    { value.j2(0U) } noexcept -> std::same_as<void>;
    { value.j3(0U) } noexcept -> std::same_as<void>;
};

template <typename T, typename Vec2, typename Vec3, typename Vec4, typename Colour3,
          typename Joints4>
concept has_nothrow_vertex_accessors = requires(T value, const T& const_value, Vec2 vec2, Vec3 vec3,
                                                Vec4 vec4, Colour3 colour, Joints4 joints) {
    { const_value.position() } noexcept -> std::same_as<Vec3>;
    { const_value.colour() } noexcept -> std::same_as<Colour3>;
    { const_value.normal() } noexcept -> std::same_as<Vec3>;
    { const_value.texCoord() } noexcept -> std::same_as<Vec2>;
    { const_value.texCoord1() } noexcept -> std::same_as<Vec2>;
    { const_value.joints() } noexcept -> std::same_as<Joints4>;
    { const_value.tangent() } noexcept -> std::same_as<Vec4>;
    { const_value.weights() } noexcept -> std::same_as<Vec4>;
    { value.position(vec3) } noexcept -> std::same_as<void>;
    { value.colour(colour) } noexcept -> std::same_as<void>;
    { value.normal(vec3) } noexcept -> std::same_as<void>;
    { value.texCoord(vec2) } noexcept -> std::same_as<void>;
    { value.texCoord1(vec2) } noexcept -> std::same_as<void>;
    { value.joints(joints) } noexcept -> std::same_as<void>;
    { value.tangent(vec4) } noexcept -> std::same_as<void>;
    { value.weights(vec4) } noexcept -> std::same_as<void>;
};

template <typename T>
concept has_nothrow_mat4_access = requires(T value, const T& const_value) {
    { const_value[0, 0] } noexcept -> std::same_as<float>;
    { value[0, 0] } noexcept -> std::same_as<float&>;
    { const_value.data() } noexcept -> std::same_as<const float*>;
    requires nothrow_assignable_from_v<decltype(std::declval<T&>()[0, 0]), float>;
};

template <typename T, typename Vec3>
concept has_nothrow_mat4_factories = requires {
    { T::identity() } noexcept -> std::same_as<T>;
    { T::rotateX(0.0f) } noexcept -> std::same_as<T>;
    { T::rotateY(0.0f) } noexcept -> std::same_as<T>;
    { T::rotateZ(0.0f) } noexcept -> std::same_as<T>;
    { T::translate(Vec3{}) } noexcept -> std::same_as<T>;
    { T::scale(Vec3{}) } noexcept -> std::same_as<T>;
    {
        T::lookAt(Vec3{}, Vec3{0.0f, 0.0f, -1.0f}, Vec3{0.0f, 1.0f, 0.0f})
    } noexcept -> std::same_as<T>;
    { T::perspective(1.0f, 1.0f, 0.1f, 100.0f) } noexcept -> std::same_as<T>;
    { T::ortho(-1.0f, 1.0f, -1.0f, 1.0f, 0.1f, 100.0f) } noexcept -> std::same_as<T>;
};

template <typename T, typename Vec4>
concept has_nothrow_mat4_arithmetic = requires(T lhs, const T& rhs, const Vec4& vec) {
    { lhs * rhs } noexcept -> std::same_as<T>;
    { lhs *= rhs } noexcept -> std::same_as<T&>;
    { lhs * vec } noexcept -> std::same_as<Vec4>;
};

template <typename T>
concept has_nothrow_animation_sampling = requires(const T& animation) {
    { animation.sample(0.0f) } noexcept;
    { animation.duration() } noexcept -> std::same_as<float>;
};

template <typename T>
concept has_nothrow_animation_state_operations = requires(T state, const T& const_state) {
    { const_state.activeAnimation() } noexcept;
    { const_state.hasActiveAnimation() } noexcept -> std::same_as<bool>;
    { state.activeAnimation(std::size_t{}) } noexcept -> std::same_as<void>;
};

template <typename T, typename Vec3>
concept has_nothrow_camera_state_operations = requires(T state, const T& const_state, Vec3 vec3) {
    { const_state.deltaPosition() } noexcept -> std::same_as<Vec3>;
    { const_state.deltaYaw() } noexcept -> std::same_as<float>;
    { const_state.deltaPitch() } noexcept -> std::same_as<float>;
    { const_state.deltaZoom() } noexcept -> std::same_as<float>;
    { const_state.time() } noexcept -> std::same_as<double>;
    { state.deltaPosition(vec3) } noexcept -> std::same_as<void>;
    { state.deltaYaw(0.0f) } noexcept -> std::same_as<void>;
    { state.deltaPitch(0.0f) } noexcept -> std::same_as<void>;
    { state.deltaZoom(0.0f) } noexcept -> std::same_as<void>;
    { state.time(0.0) } noexcept -> std::same_as<void>;
};

template <typename T, typename Vec3>
concept has_nothrow_controller_state_operations =
    requires(T state, const T& const_state, Vec3 vec3) {
        { const_state.deltaPosition() } noexcept -> std::same_as<Vec3>;
        { const_state.time() } noexcept -> std::same_as<double>;
        { state.deltaPosition(vec3) } noexcept -> std::same_as<void>;
        { state.time(0.0) } noexcept -> std::same_as<void>;
    };

template <typename T>
concept has_nothrow_variant_state_operations = requires(T state, const T& const_state) {
    { const_state.cycleDelta() } noexcept -> std::same_as<int>;
    { const_state.hasCycleCommand() } noexcept -> std::same_as<bool>;
    { state.cycleDelta(0) } noexcept -> std::same_as<void>;
};

template <typename T, typename CameraState, typename AnimationState, typename ControllerState,
          typename VariantState>
concept has_nothrow_input_state_operations =
    requires(T state, const T& const_state, CameraState camera, AnimationState animation,
             ControllerState controller, VariantState variant) {
        { state.cameraState() } noexcept -> std::same_as<CameraState&>;
        { const_state.cameraState() } noexcept -> std::same_as<const CameraState&>;
        { state.cameraState(camera) } noexcept -> std::same_as<void>;
        { state.animationState() } noexcept -> std::same_as<AnimationState&>;
        { const_state.animationState() } noexcept -> std::same_as<const AnimationState&>;
        { state.animationState(animation) } noexcept -> std::same_as<void>;
        { state.controllerState() } noexcept -> std::same_as<ControllerState&>;
        { const_state.controllerState() } noexcept -> std::same_as<const ControllerState&>;
        { state.controllerState(controller) } noexcept -> std::same_as<void>;
        { state.variantState() } noexcept -> std::same_as<VariantState&>;
        { const_state.variantState() } noexcept -> std::same_as<const VariantState&>;
        { state.variantState(variant) } noexcept -> std::same_as<void>;
        { const_state.deltaTime() } noexcept -> std::same_as<float>;
        { state.deltaTime(0.0f) } noexcept -> std::same_as<void>;
        { const_state.time() } noexcept -> std::same_as<double>;
        { state.time(0.0) } noexcept -> std::same_as<void>;
    };

} // namespace test_traits
