/*
Uniform-sphere-direction counterpart to rdv's own XREnvironmentSampler
(rdv/shaders/rendering/xr_environment_sampler.h) - same Map[6, 3+SPECTRAL_
DIM+1] interface (direction(3) | env(dir)*weight(SPECTRAL_DIM) | pdf(1)),
but for a constant ambient color instead of an image: every direction is
equally likely (pdf = 1/4pi over the full sphere), so the returned
"env(dir)*weight" term is just color*4pi (weight = 1/pdf = 4pi) - no
quadtree, no image lookup, same alpha/uniform-by-solid-angle beta sampling
tail end xr_environment_sampler.h itself uses for its own leaf-pixel
sampling, just over the whole sphere instead of one leaf.
*/
void forward(RDV_NODE node, float _input[IN], out float _output[OUT])
{
    float alpha = mix(-pi, pi, random());
    float s = mix(-1.0, 1.0, random()); // sin(beta) uniform -> uniform-by-solid-angle beta
    float beta = asin(clamp(s, -1.0, 1.0));
    vec3 dir = usc2dir(vec2(alpha, beta));

    float pdf = 1.0 / (4.0 * pi);
    float weight = 4.0 * pi; // 1/pdf

    vec3 color = deref(node).color;
    _output[0] = dir.x;
    _output[1] = dir.y;
    _output[2] = dir.z;
    _output[3] = color.x * weight;
    _output[4] = color.y * weight;
    _output[5] = color.z * weight;
    _output[6] = pdf;
}
