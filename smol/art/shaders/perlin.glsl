float hash31(vec3 p) {
    // cheap-ish hash
    p = fract(p * 0.1031);
    p += dot(p, p.yzx + 33.33);
    return fract((p.x + p.y) * p.z);
}

float noise3(vec3 p) {
    // value noise (trilinear interpolation)
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f*f*(3.0 - 2.0*f);

    float n000 = hash31(i + vec3(0,0,0));
    float n100 = hash31(i + vec3(1,0,0));
    float n010 = hash31(i + vec3(0,1,0));
    float n110 = hash31(i + vec3(1,1,0));
    float n001 = hash31(i + vec3(0,0,1));
    float n101 = hash31(i + vec3(1,0,1));
    float n011 = hash31(i + vec3(0,1,1));
    float n111 = hash31(i + vec3(1,1,1));

    float nx00 = mix(n000, n100, f.x);
    float nx10 = mix(n010, n110, f.x);
    float nx01 = mix(n001, n101, f.x);
    float nx11 = mix(n011, n111, f.x);

    float nxy0 = mix(nx00, nx10, f.y);
    float nxy1 = mix(nx01, nx11, f.y);

    return mix(nxy0, nxy1, f.z);
}

float fbm(vec3 p) {
    float a = 0.5;
    float s = 0.0;
    for (int i = 0; i < 5; i++) {
        s += a * noise3(p);
        p *= 2.02;
        a *= 0.5;
    }
    return s; // ~[0,1]
}
float evaluate_sdf(in vec3 point) {
    const vec3 center = vec3(0.0, -40.0, 5.0);
    const float radius = 2.0;

    // time envelope: 0 -> 1 -> 0 (period seconds)
    const float period = 4.0;
    float phase = 6.2831853 * (iTime / period);
    float amp = 0.5 - 0.5 * cos(phase);          // [0,1]
    amp = smoothstep(0.0, 1.0, amp);             // softer easing

    // deformation: low-frequency + time advection
    vec3 p = point - center;
    float n = fbm(p * 1.2 + vec3(0.0, 0.6 * iTime, 0.0)); // [0,1]
    n = (n - 0.5) * 2.0;                                   // [-1,1]

    // bias deformation to the surface (optional: makes interior less weird)
    float r = length(p);
    float surface_mask = exp(-6.0 * abs(r - radius));      // ~1 near surface
    float disp = 0.6 * n * surface_mask;                   // signed displacement

    float deformed_radius = radius + amp * disp;
    return r - deformed_radius;
}

vec3 ray_marching(in vec3 ray_origin, in vec3 ray_direction, in vec3 light_source) {
    const int num_steps = 100;
    const float hit_threshold = 0.001;
    float sdf = 0.0;
    int object_index = -1;
    const vec3 perturbation = vec3(0.001, 0.0, 0.0);
    vec3 normal;
    float d = 0.0;

    for (int i = 0; i < num_steps; i++) {
        // for each step, get distance to objects in scene
        vec3 point = ray_origin + d * ray_direction;
        float sdf = evaluate_sdf(point);
        // if distance is below threshold, compute color and return
        if (sdf < hit_threshold) {
            vec3 normal = normalize(
                vec3(
                    evaluate_sdf(point + perturbation.xyz) - evaluate_sdf(point - perturbation.xyz),
                    evaluate_sdf(point + perturbation.yxz) - evaluate_sdf(point - perturbation.yxz),
                    evaluate_sdf(point + perturbation.yzx) - evaluate_sdf(point - perturbation.yzx)
                )
            );
            vec3 point_to_light = normalize(light_source - point);
            float intensity = max(0.0, dot(point_to_light, normal));
            return vec3(0.0, intensity, 0.0);
        }
        // else take a step of size sdf ... see https://michaelwalczyk.com/blog-ray-marching.html
        d += sdf;

    }
    return vec3(0.0, 0.0, 0.0); // black
}


void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    float focal_length = 5.0;
    vec3 camera_position = vec3(0, 0, 0);
    vec3 screen_center = vec3(0, -focal_length, 0);


    const float oscillation_period_s = 5.0;
    float angle_radians = iTime * 2.0 * 3.14 / (oscillation_period_s);
    vec3 light_source = vec3(
        5.0 * focal_length * cos(angle_radians),
        -5.0 * focal_length,
        2.0 * focal_length * sin(angle_radians)
    );
    // shoot a ray from camera_position to screen_point ~= fragCoord
    vec2 screen_point_xy = (2.0*fragCoord-iResolution.xy)/iResolution.y;
    vec3 screen_point = vec3(screen_point_xy.x, 0.0, screen_point_xy.y) + screen_center;

    // https://registry.khronos.org/OpenGL-Refpages/gl4/html/normalize.xhtml
    vec3 ray_direction = normalize(screen_point - camera_position);
    vec3 ray_origin = camera_position;

    // TODO: also pass in a scene or something?
    vec3 col = ray_marching(ray_origin, ray_direction, light_source);
    // Output to screen
    fragColor = vec4(col,1.0);
}