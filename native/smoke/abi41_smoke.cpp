#include "ssbl_xpbd_cuda.h"
#include "ssbl_abi41_cuda_types.cuh"

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

bool finite_positions(const std::vector<float>& values) {
    for (float value : values) {
        if (!std::isfinite(value)) {
            return false;
        }
    }
    return true;
}

float distance3(const std::vector<float>& values, int a, int b) {
    const int ia = a * 3;
    const int ib = b * 3;
    const float dx = values[ia + 0] - values[ib + 0];
    const float dy = values[ia + 1] - values[ib + 1];
    const float dz = values[ia + 2] - values[ib + 2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

float edge_average_z(const std::vector<float>& values, int a, int b) {
    return (values[a * 3 + 2] + values[b * 3 + 2]) * 0.5f;
}

int run_self_collision_smoke() {
    constexpr int kVertexCount = 600;
    constexpr int kPairVertex = kVertexCount - 1;
    std::vector<float> rest_positions(kVertexCount * 3, 0.0f);
    std::vector<float> inv_mass(kVertexCount, 1.0f);
    for (int i = 0; i < kVertexCount; ++i) {
        rest_positions[i * 3 + 0] = static_cast<float>(i);
    }

    SsblXpbdConfig cfg{};
    cfg.vertex_count = kVertexCount;
    cfg.dt = 1.0f / 60.0f;
    cfg.damping = 1.0f;
    cfg.gravity[0] = 0.0f;
    cfg.gravity[1] = 0.0f;
    cfg.gravity[2] = 0.0f;
    cfg.self_collision = 1;
    cfg.self_collision_mode = 1;
    cfg.cloth_thickness = 0.10f;
    cfg.collision_margin = 0.0f;
    cfg.max_self_collision_neighbors = 32;

    SsblXpbdMesh mesh{};
    mesh.positions = rest_positions.data();
    mesh.inv_mass = inv_mass.data();

    void* solver = ssbl_create_solver(&cfg, &mesh);
    if (!solver) {
        std::fprintf(stderr, "SSBL_ABI41_SELF_COLLISION_ERROR create: %s\n", ssbl_last_error());
        return 20;
    }

    std::vector<float> close_positions = rest_positions;
    close_positions[kPairVertex * 3 + 0] = 0.03f;
    const float initial_gap = distance3(close_positions, 0, kPairVertex);
    if (!ssbl_update_positions(solver, close_positions.data(), static_cast<int>(close_positions.size()))) {
        std::fprintf(stderr, "SSBL_ABI41_SELF_COLLISION_ERROR upload: %s\n", ssbl_last_error());
        ssbl_destroy_solver(solver);
        return 21;
    }
    if (!ssbl_step_solver_ex(solver, 1, 6, 1, 1)) {
        std::fprintf(stderr, "SSBL_ABI41_SELF_COLLISION_ERROR step: %s\n", ssbl_last_error());
        ssbl_destroy_solver(solver);
        return 22;
    }

    std::vector<float> out(close_positions.size(), 0.0f);
    if (!ssbl_download_positions(solver, out.data(), static_cast<int>(out.size()))) {
        std::fprintf(stderr, "SSBL_ABI41_SELF_COLLISION_ERROR download: %s\n", ssbl_last_error());
        ssbl_destroy_solver(solver);
        return 23;
    }
    SsblXpbdDiagnostics diag{};
    if (!ssbl_get_diagnostics(solver, &diag)) {
        std::fprintf(stderr, "SSBL_ABI41_SELF_COLLISION_ERROR diagnostics: %s\n", ssbl_last_error());
        ssbl_destroy_solver(solver);
        return 24;
    }
    ssbl_destroy_solver(solver);

    const float final_gap = distance3(out, 0, kPairVertex);
    if (!finite_positions(out) || !diag.finite_flag) {
        std::fprintf(stderr, "SSBL_ABI41_SELF_COLLISION_ERROR non-finite output\n");
        return 25;
    }
    if (diag.abi41_soft_contact_count + diag.abi41_exact_impulse_contact_count <= 0
        || diag.fast_soft_repulsion_candidates <= 0) {
        std::fprintf(stderr, "SSBL_ABI41_SELF_COLLISION_ERROR no soft contacts\n");
        return 26;
    }
    if (!(diag.abi41_max_smoothed_delta > 0.0f)) {
        std::fprintf(stderr, "SSBL_ABI41_SELF_COLLISION_ERROR no smoothed delta\n");
        return 28;
    }
    if (!(final_gap > initial_gap + 1.0e-4f)) {
        std::fprintf(
            stderr,
            "SSBL_ABI41_SELF_COLLISION_ERROR gap did not increase initial=%.6f final=%.6f\n",
            initial_gap,
            final_gap
        );
        return 27;
    }
    std::printf(
        "SSBL_ABI41_SELF_COLLISION_OK vertices=%d soft=%lld exact=%lld candidates=%lld max_delta=%.5f gap=%.5f->%.5f\n",
        kVertexCount,
        diag.abi41_soft_contact_count,
        diag.abi41_exact_impulse_contact_count,
        diag.fast_soft_repulsion_candidates,
        diag.abi41_max_smoothed_delta,
        initial_gap,
        final_gap
    );
    return 0;
}

int run_self_vertex_triangle_smoke() {
    constexpr int kVertexCount = 600;
    constexpr int kProbeVertex = kVertexCount - 1;
    constexpr int kTriangleCount = 64;
    std::vector<float> rest_positions(kVertexCount * 3, 0.0f);
    std::vector<float> inv_mass(kVertexCount, 1.0f);
    rest_positions[0] = -1.0f;
    rest_positions[1] = -1.0f;
    rest_positions[2] = 0.0f;
    rest_positions[3] = 1.0f;
    rest_positions[4] = -1.0f;
    rest_positions[5] = 0.0f;
    rest_positions[6] = 0.0f;
    rest_positions[7] = 1.0f;
    rest_positions[8] = 0.0f;
    inv_mass[0] = 0.0f;
    inv_mass[1] = 0.0f;
    inv_mass[2] = 0.0f;
    for (int i = 3; i < kVertexCount; ++i) {
        rest_positions[i * 3 + 0] = 20.0f + static_cast<float>(i);
        rest_positions[i * 3 + 1] = 20.0f + static_cast<float>(i % 7) * 0.25f;
        rest_positions[i * 3 + 2] = static_cast<float>(i % 5) * 0.20f;
    }
    rest_positions[kProbeVertex * 3 + 0] = 5.0f;
    rest_positions[kProbeVertex * 3 + 1] = 5.0f;
    rest_positions[kProbeVertex * 3 + 2] = 1.0f;

    std::vector<int> triangles;
    triangles.reserve(kTriangleCount * 3);
    triangles.insert(triangles.end(), {0, 1, 2});
    for (int t = 1; t < kTriangleCount; ++t) {
        const int base = 3 + t * 3;
        triangles.push_back(base + 0);
        triangles.push_back(base + 1);
        triangles.push_back(base + 2);
    }

    SsblXpbdConfig cfg{};
    cfg.vertex_count = kVertexCount;
    cfg.triangle_count = kTriangleCount;
    cfg.dt = 1.0f / 60.0f;
    cfg.damping = 1.0f;
    cfg.gravity[0] = 0.0f;
    cfg.gravity[1] = 0.0f;
    cfg.gravity[2] = 0.0f;
    cfg.self_collision = 1;
    cfg.self_collision_mode = 1;
    cfg.cloth_thickness = 0.10f;
    cfg.collision_margin = 0.0f;
    cfg.max_self_collision_neighbors = 32;

    SsblXpbdMesh mesh{};
    mesh.positions = rest_positions.data();
    mesh.inv_mass = inv_mass.data();
    mesh.triangles = triangles.data();

    void* solver = ssbl_create_solver(&cfg, &mesh);
    if (!solver) {
        std::fprintf(stderr, "SSBL_ABI41_SELF_VT_ERROR create: %s\n", ssbl_last_error());
        return 30;
    }

    std::vector<float> close_positions = rest_positions;
    close_positions[kProbeVertex * 3 + 0] = 0.0f;
    close_positions[kProbeVertex * 3 + 1] = -0.2f;
    close_positions[kProbeVertex * 3 + 2] = 0.03f;
    const float initial_height = close_positions[kProbeVertex * 3 + 2];
    if (!ssbl_update_positions(solver, close_positions.data(), static_cast<int>(close_positions.size()))) {
        std::fprintf(stderr, "SSBL_ABI41_SELF_VT_ERROR upload: %s\n", ssbl_last_error());
        ssbl_destroy_solver(solver);
        return 31;
    }
    if (!ssbl_step_solver_ex(solver, 1, 6, 1, 1)) {
        std::fprintf(stderr, "SSBL_ABI41_SELF_VT_ERROR step: %s\n", ssbl_last_error());
        ssbl_destroy_solver(solver);
        return 32;
    }

    std::vector<float> out(close_positions.size(), 0.0f);
    if (!ssbl_download_positions(solver, out.data(), static_cast<int>(out.size()))) {
        std::fprintf(stderr, "SSBL_ABI41_SELF_VT_ERROR download: %s\n", ssbl_last_error());
        ssbl_destroy_solver(solver);
        return 33;
    }
    SsblXpbdDiagnostics diag{};
    if (!ssbl_get_diagnostics(solver, &diag)) {
        std::fprintf(stderr, "SSBL_ABI41_SELF_VT_ERROR diagnostics: %s\n", ssbl_last_error());
        ssbl_destroy_solver(solver);
        return 34;
    }
    ssbl_destroy_solver(solver);

    const float final_height = out[kProbeVertex * 3 + 2];
    if (!finite_positions(out) || !diag.finite_flag) {
        std::fprintf(stderr, "SSBL_ABI41_SELF_VT_ERROR non-finite output\n");
        return 35;
    }
    if (diag.abi41_soft_contact_count <= 0 || diag.fast_soft_repulsion_candidates <= 0) {
        std::fprintf(stderr, "SSBL_ABI41_SELF_VT_ERROR no VT soft contacts\n");
        return 36;
    }
    if (!(diag.abi41_max_smoothed_delta > 0.0f)) {
        std::fprintf(stderr, "SSBL_ABI41_SELF_VT_ERROR no smoothed delta\n");
        return 38;
    }
    if (!(final_height > initial_height + 1.0e-4f)) {
        std::fprintf(
            stderr,
            "SSBL_ABI41_SELF_VT_ERROR height did not increase initial=%.6f final=%.6f\n",
            initial_height,
            final_height
        );
        return 37;
    }
    std::printf(
        "SSBL_ABI41_SELF_VT_OK vertices=%d triangles=%d soft=%lld exact=%lld candidates=%lld max_delta=%.5f height=%.5f->%.5f\n",
        kVertexCount,
        kTriangleCount,
        diag.abi41_soft_contact_count,
        diag.abi41_exact_impulse_contact_count,
        diag.fast_soft_repulsion_candidates,
        diag.abi41_max_smoothed_delta,
        initial_height,
        final_height
    );
    return 0;
}

int run_self_edge_edge_smoke() {
    constexpr int kVertexCount = 600;
    constexpr int kEdgeCount = 64;
    std::vector<float> rest_positions(kVertexCount * 3, 0.0f);
    std::vector<float> inv_mass(kVertexCount, 1.0f);
    for (int i = 0; i < kVertexCount; ++i) {
        rest_positions[i * 3 + 0] = 200.0f + static_cast<float>(i) * 3.0f;
        rest_positions[i * 3 + 1] = 100.0f + static_cast<float>(i % 11);
        rest_positions[i * 3 + 2] = 0.0f;
    }

    rest_positions[0] = -1.0f;
    rest_positions[1] = 0.0f;
    rest_positions[2] = 0.0f;
    rest_positions[3] = 1.0f;
    rest_positions[4] = 0.0f;
    rest_positions[5] = 0.0f;
    rest_positions[6] = 5.0f;
    rest_positions[7] = -1.0f;
    rest_positions[8] = 0.0f;
    rest_positions[9] = 5.0f;
    rest_positions[10] = 1.0f;
    rest_positions[11] = 0.0f;
    inv_mass[0] = 0.0f;
    inv_mass[1] = 0.0f;

    std::vector<int> edges;
    edges.reserve(kEdgeCount * 2);
    edges.insert(edges.end(), {0, 1, 2, 3});
    for (int e = 2; e < kEdgeCount; ++e) {
        const int a = 4 + (e - 2) * 2;
        const int b = a + 1;
        rest_positions[a * 3 + 0] = 50.0f + static_cast<float>(e) * 4.0f;
        rest_positions[a * 3 + 1] = 30.0f;
        rest_positions[a * 3 + 2] = 0.0f;
        rest_positions[b * 3 + 0] = 51.0f + static_cast<float>(e) * 4.0f;
        rest_positions[b * 3 + 1] = 30.0f;
        rest_positions[b * 3 + 2] = 0.0f;
        edges.push_back(a);
        edges.push_back(b);
    }

    std::vector<float> rest_lengths;
    rest_lengths.reserve(kEdgeCount);
    for (int e = 0; e < kEdgeCount; ++e) {
        const int a = edges[e * 2 + 0] * 3;
        const int b = edges[e * 2 + 1] * 3;
        const float dx = rest_positions[a + 0] - rest_positions[b + 0];
        const float dy = rest_positions[a + 1] - rest_positions[b + 1];
        const float dz = rest_positions[a + 2] - rest_positions[b + 2];
        rest_lengths.push_back(std::sqrt(dx * dx + dy * dy + dz * dz));
    }

    SsblXpbdConfig cfg{};
    cfg.vertex_count = kVertexCount;
    cfg.edge_count = kEdgeCount;
    cfg.dt = 1.0f / 60.0f;
    cfg.damping = 1.0f;
    cfg.gravity[0] = 0.0f;
    cfg.gravity[1] = 0.0f;
    cfg.gravity[2] = 0.0f;
    cfg.self_collision = 1;
    cfg.self_collision_mode = 1;
    cfg.cloth_thickness = 0.10f;
    cfg.collision_margin = 0.0f;
    cfg.max_self_collision_neighbors = 32;

    SsblXpbdMesh mesh{};
    mesh.positions = rest_positions.data();
    mesh.inv_mass = inv_mass.data();
    mesh.edges = edges.data();
    mesh.edge_rest_lengths = rest_lengths.data();

    void* solver = ssbl_create_solver(&cfg, &mesh);
    if (!solver) {
        std::fprintf(stderr, "SSBL_ABI41_SELF_EE_ERROR create: %s\n", ssbl_last_error());
        return 40;
    }

    std::vector<float> close_positions = rest_positions;
    close_positions[6] = 0.0f;
    close_positions[7] = -1.0f;
    close_positions[8] = 0.03f;
    close_positions[9] = 0.0f;
    close_positions[10] = 1.0f;
    close_positions[11] = 0.03f;
    const float initial_height = edge_average_z(close_positions, 2, 3);
    if (!ssbl_update_positions(solver, close_positions.data(), static_cast<int>(close_positions.size()))) {
        std::fprintf(stderr, "SSBL_ABI41_SELF_EE_ERROR upload: %s\n", ssbl_last_error());
        ssbl_destroy_solver(solver);
        return 41;
    }
    if (!ssbl_step_solver_ex(solver, 1, 6, 1, 1)) {
        std::fprintf(stderr, "SSBL_ABI41_SELF_EE_ERROR step: %s\n", ssbl_last_error());
        ssbl_destroy_solver(solver);
        return 42;
    }

    std::vector<float> out(close_positions.size(), 0.0f);
    if (!ssbl_download_positions(solver, out.data(), static_cast<int>(out.size()))) {
        std::fprintf(stderr, "SSBL_ABI41_SELF_EE_ERROR download: %s\n", ssbl_last_error());
        ssbl_destroy_solver(solver);
        return 43;
    }
    SsblXpbdDiagnostics diag{};
    if (!ssbl_get_diagnostics(solver, &diag)) {
        std::fprintf(stderr, "SSBL_ABI41_SELF_EE_ERROR diagnostics: %s\n", ssbl_last_error());
        ssbl_destroy_solver(solver);
        return 44;
    }
    ssbl_destroy_solver(solver);

    const float final_height = edge_average_z(out, 2, 3);
    if (!finite_positions(out) || !diag.finite_flag) {
        std::fprintf(stderr, "SSBL_ABI41_SELF_EE_ERROR non-finite output\n");
        return 45;
    }
    if (diag.abi41_edge_edge_contact_count <= 0 || diag.resolved_contacts <= 0) {
        std::fprintf(stderr, "SSBL_ABI41_SELF_EE_ERROR no EE contacts\n");
        return 46;
    }
    if (!(diag.abi41_max_smoothed_delta > 0.0f)) {
        std::fprintf(stderr, "SSBL_ABI41_SELF_EE_ERROR no smoothed delta\n");
        return 48;
    }
    if (!(final_height > initial_height + 1.0e-4f)) {
        std::fprintf(
            stderr,
            "SSBL_ABI41_SELF_EE_ERROR height did not increase initial=%.6f final=%.6f contacts=%lld\n",
            initial_height,
            final_height,
            diag.abi41_edge_edge_contact_count
        );
        return 47;
    }
    std::printf(
        "SSBL_ABI41_SELF_EE_OK vertices=%d edges=%d edge_contacts=%lld candidates=%lld max_delta=%.5f height=%.5f->%.5f\n",
        kVertexCount,
        kEdgeCount,
        diag.abi41_edge_edge_contact_count,
        diag.fast_soft_repulsion_candidates,
        diag.abi41_max_smoothed_delta,
        initial_height,
        final_height
    );
    return 0;
}

} // namespace

int main() {
    static_assert(sizeof(ssbl_abi41::float3) == 12);
    static_assert(sizeof(ssbl_abi41::CudaSpringPBD) == 12);
    static_assert(sizeof(ssbl_abi41::CudaTriangle) == 12);
    static_assert(sizeof(ssbl_abi41::symMatCuda) == 24);

    std::vector<float> positions = {
        0.0f, 0.2f, 0.0f,
        1.0f, 0.2f, 0.0f,
        0.0f, 1.2f, 0.0f,
        1.0f, 1.2f, 0.0f,
    };
    std::vector<float> inv_mass = {0.0f, 0.0f, 1.0f, 1.0f};
    std::vector<int> edges = {
        0, 1,
        0, 2,
        1, 3,
        2, 3,
        0, 3,
        1, 2,
    };
    std::vector<float> rest_lengths;
    rest_lengths.reserve(edges.size() / 2);
    for (size_t e = 0; e < edges.size(); e += 2) {
        int a = edges[e] * 3;
        int b = edges[e + 1] * 3;
        float dx = positions[a + 0] - positions[b + 0];
        float dy = positions[a + 1] - positions[b + 1];
        float dz = positions[a + 2] - positions[b + 2];
        rest_lengths.push_back(std::sqrt(dx * dx + dy * dy + dz * dz));
    }
    std::vector<int> triangles = {0, 1, 2, 1, 3, 2};

    SsblXpbdConfig cfg{};
    cfg.vertex_count = 4;
    cfg.edge_count = static_cast<int>(rest_lengths.size());
    cfg.triangle_count = 2;
    cfg.dt = 1.0f / 60.0f;
    cfg.damping = 0.99f;
    cfg.gravity[0] = 0.0f;
    cfg.gravity[1] = -9.8f;
    cfg.gravity[2] = 0.0f;
    cfg.stretch_compliance = 1.0e-6f;
    cfg.collision_margin = 0.005f;
    cfg.use_ground = 1;
    cfg.ground_height = 0.0f;
    cfg.cloth_thickness = 0.02f;

    SsblXpbdMesh mesh{};
    mesh.positions = positions.data();
    mesh.inv_mass = inv_mass.data();
    mesh.edges = edges.data();
    mesh.edge_rest_lengths = rest_lengths.data();
    mesh.triangles = triangles.data();

    void* solver = ssbl_create_solver(&cfg, &mesh);
    if (!solver) {
        std::fprintf(stderr, "SSBL_ABI41_NATIVE_ERROR create: %s\n", ssbl_last_error());
        return 2;
    }
    int pins[] = {0, 1};
    float pin_positions[] = {
        0.0f, 0.2f, 0.0f,
        1.0f, 0.2f, 0.0f,
    };
    if (!ssbl_update_pin_targets(solver, pins, pin_positions, 2)) {
        std::fprintf(stderr, "SSBL_ABI41_NATIVE_ERROR pins: %s\n", ssbl_last_error());
        ssbl_destroy_solver(solver);
        return 3;
    }
    if (!ssbl_step_solver_ex(solver, 4, 4, 1, 1)) {
        std::fprintf(stderr, "SSBL_ABI41_NATIVE_ERROR step: %s\n", ssbl_last_error());
        ssbl_destroy_solver(solver);
        return 4;
    }
    std::vector<float> out(positions.size(), 0.0f);
    if (!ssbl_download_positions(solver, out.data(), static_cast<int>(out.size()))) {
        std::fprintf(stderr, "SSBL_ABI41_NATIVE_ERROR download: %s\n", ssbl_last_error());
        ssbl_destroy_solver(solver);
        return 5;
    }
    SsblXpbdDiagnostics diag{};
    if (!ssbl_get_diagnostics(solver, &diag)) {
        std::fprintf(stderr, "SSBL_ABI41_NATIVE_ERROR diagnostics: %s\n", ssbl_last_error());
        ssbl_destroy_solver(solver);
        return 6;
    }
    ssbl_destroy_solver(solver);
    if (!finite_positions(out) || !diag.finite_flag) {
        std::fprintf(stderr, "SSBL_ABI41_NATIVE_ERROR non-finite output\n");
        return 7;
    }
    for (int pin = 0; pin < 2; ++pin) {
        const int vertex = pins[pin];
        for (int axis = 0; axis < 3; ++axis) {
            const float actual = out[vertex * 3 + axis];
            const float expected = pin_positions[pin * 3 + axis];
            if (std::fabs(actual - expected) > 1.0e-5f) {
                std::fprintf(stderr, "SSBL_ABI41_NATIVE_ERROR pin target mismatch\n");
                return 8;
            }
        }
    }
    if (!(out[7] < positions[7])) {
        std::fprintf(stderr, "SSBL_ABI41_NATIVE_ERROR cloth did not move under gravity\n");
        return 9;
    }
    std::printf(
        "SSBL_ABI41_NATIVE_OK vertices=%d step_ms=%.3f contacts=%lld\n",
        cfg.vertex_count,
        diag.step_ms,
        diag.resolved_contacts
    );
    int self_result = run_self_collision_smoke();
    if (self_result != 0) {
        return self_result;
    }
    int vt_result = run_self_vertex_triangle_smoke();
    if (vt_result != 0) {
        return vt_result;
    }
    return run_self_edge_edge_smoke();
}
