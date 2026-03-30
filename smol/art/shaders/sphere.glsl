
float evaluate_sdf(in vec3 point) {
    // only 1 sphere
    const vec3 center = vec3(0, -40, 5.0);
    const float radius = 2.0;
    return length(point - center) - radius;
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