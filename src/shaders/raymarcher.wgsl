const THREAD_COUNT = 16;
const PI = 3.1415927f;
const MAX_DIST = 1000.0;

@group(0) @binding(0)  
  var<storage, read_write> fb : array<vec4f>;

@group(1) @binding(0)
  var<storage, read_write> uniforms : array<f32>;

@group(2) @binding(0)
  var<storage, read_write> shapesb : array<shape>;

@group(2) @binding(1)
  var<storage, read_write> shapesinfob : array<vec4f>;

struct shape {
  transform : vec4f, // xyz = position
  radius : vec4f, // xyz = scale, w = global scale
  rotation : vec4f, // xyz = rotation
  op : vec4f, // x = operation, y = k value, z = repeat mode, w = repeat offset
  color : vec4f, // xyz = color
  animate_transform : vec4f, // xyz = animate position value (sin amplitude), w = animate speed
  animate_rotation : vec4f, // xyz = animate rotation value (sin amplitude), w = animate speed
  quat : vec4f, // xyzw = quaternion
  transform_animated : vec4f, // xyz = position buffer
};

struct march_output {
  color : vec3f,
  depth : f32,
  outline : bool,
};

fn op_smooth_union(d1: f32, d2: f32, col1: vec3f, col2: vec3f, k: f32) -> vec4f
{
  var h = clamp(0.5 + 0.5 * (d2 - d1) / max(k, 0.0001), 0.0, 1.0);
  var dist = mix(d2, d1, h) - k * h * (1.0 - h);
  var color = mix(col2, col1, h);
  return vec4f(color, dist);
}

fn op_smooth_subtraction(d1: f32, d2: f32, col1: vec3f, col2: vec3f, k: f32) -> vec4f
{
  var h = clamp(0.5 - 0.5 * (d2 + d1) / max(k, 0.0001), 0.0, 1.0);
  var dist = mix(d2, -d1, h) + k * h * (1.0 - h);
  var color = mix(col2, col1, h);
  return vec4f(color, dist);
}

fn op_smooth_intersection(d1: f32, d2: f32, col1: vec3f, col2: vec3f, k: f32) -> vec4f
{
  var h = clamp(0.5 - 0.5 * (d2 - d1) / max(k, 0.0001), 0.0, 1.0);
  var dist = mix(d2, d1, h) + k * h * (1.0 - h);
  var color = mix(col2, col1, h);
  return vec4f(color, dist);
}

fn op(op: f32, d1: f32, d2: f32, col1: vec3f, col2: vec3f, k: f32) -> vec4f
{
  // union
  if (op < 1.0)
  {
    return op_smooth_union(d1, d2, col1, col2, k);
  }

  // subtraction
  if (op < 2.0)
  {
    return op_smooth_subtraction(d2, d1, col2, col1, k);
  }

  // intersection
  return op_smooth_intersection(d2, d1, col2, col1, k);
}

fn repeat(p: vec3f, offset: vec3f) -> vec3f
{
  if (all(offset == vec3f(0.0))) { // Verificar se offset é zero
        return p; // Sem repetição
  }
  return modc(p + 0.5 * offset, offset) - 0.5 * offset;
}

fn transform_p(p: vec3f, option: vec2f) -> vec3f
{
  // normal mode
  if (option.x <= 1.0)
  {
    return p;
  }

  // return repeat / mod mode
  return repeat(p, vec3f(option.y));
}

fn scene(p: vec3f) -> vec4f // xyz = color, w = distance
{
    var d = mix(100.0, p.y, uniforms[17]);

    var spheresCount = i32(uniforms[2]);
    var boxesCount = i32(uniforms[3]);
    var torusCount = i32(uniforms[4]);

    var all_objects_count = spheresCount + boxesCount + torusCount;
    var result = vec4f(vec3f(1.0), d);

    for (var i = 0; i < all_objects_count; i = i + 1) {
        var shape_info = shapesinfob[i];
        var shape_type = f32(shape_info.x);
        var shape_index = i32(shape_info.y);
        var shape = shapesb[shape_index];
        var animated_transform = shape.animate_transform.xyz * sin(uniforms[0] * shape.animate_transform.w);

        var transformed_p = p - (shape.transform.xyz + animated_transform);
        transformed_p = transform_p(transformed_p, shape.op.zw);
        var animated_rotation = shape.animate_rotation.xyz * sin(uniforms[0] * shape.animate_rotation.w);
        var quat_animated = quaternion_from_euler(animated_rotation + shape.rotation.xyz);
        var dist: f32;
        if (shape_type > 1) {

            dist = sdf_torus(transformed_p, shape.radius.xy, quat_animated);
            
        } else if (shape_type > 0) {

            dist = sdf_round_box(transformed_p, shape.radius.xyz, shape.radius.w, quat_animated);
        } else {

            dist = sdf_sphere(transformed_p, shape.radius, quat_animated);
        } 
        if (dist < result.w) {
            result.w = dist; // assign closest distance
            let res = vec4f(shape.color.xyz,dist);
            result = res; // assign color and distance 
        }
        if (i > 0) { 
            result = op(shape.op.x, result.w, dist, result.xyz, shape.color.xyz, shape.op.y);
        } else if (dist < result.w) {
            result = vec4f(shape.color.xyz, dist);
        }

    }

    return result;
}

fn march(ro: vec3f, rd: vec3f) -> march_output
{
  var max_marching_steps = i32(uniforms[5]);
  var EPSILON = uniforms[23];


  var depth = 0.0;
  var color = vec3f(1.0);
  var march_step = uniforms[22];

  var hit = false;
  
  for (var i = 0; i < max_marching_steps; i = i + 1)
  {
      // Calcula a posição atual ao longo do raio
      let p = ro + rd * depth;

      // Chama a função scene para determinar a menor distância ao objeto mais próximo
      let scene_result = scene(p);
      let dist = scene_result.w;  // Distância retornada pela cena
      let obj_color = scene_result.xyz; // Cor do objeto mais próximo

      // Verifica colisão (distância menor que EPSILON)
      if (dist < EPSILON)
      {
          color = obj_color; // Atualiza a cor com a do objeto
          hit = true;        // Marca que houve uma colisão
          break;
      }

      // Incrementa a profundidade acumulada
      depth += dist ;

      // Sai do laço se a profundidade exceder o limite
      if (depth > MAX_DIST)
      {
          break;
      }
  }

  return march_output(color, depth, false);
}

fn get_normal(p: vec3f) -> vec3f
{
    let eps = 0.001; // Pequena diferença para calcular o gradiente

    // Calcula o gradiente usando a função `scene` para avaliar as distâncias
    let dx = scene(p + vec3f(eps, 0.0, 0.0)).w - scene(p - vec3f(eps, 0.0, 0.0)).w;
    let dy = scene(p + vec3f(0.0, eps, 0.0)).w - scene(p - vec3f(0.0, eps, 0.0)).w;
    let dz = scene(p + vec3f(0.0, 0.0, eps)).w - scene(p - vec3f(0.0, 0.0, eps)).w;

    // Gradiente é o vetor normal da superfície
    let grad = vec3f(dx, dy, dz);

    // Retorna o vetor normalizado
    return normalize(grad);
}

// https://iquilezles.org/articles/rmshadows/
fn get_soft_shadow(ro: vec3f, rd: vec3f, tmin: f32, tmax: f32, k: f32) -> f32
{
    var res = 1.0;
    var tempo = tmin;
    let stps = 100;

    for (var i = 0; i < stps && tempo < tmax; i = i + 1)
    {
        let p = ro + rd * tempo;
        let h = scene(p).w;
        if (h < 0.001)
        {
            return 0.0; // In shadow
        }
        res = min(res, k * h / tempo);
        tempo += h;
    }

    return clamp(res, 0.0, 1.0);
}

fn get_AO(current: vec3f, normal: vec3f) -> f32
{
  var occ = 0.0;
  var sca = 1.0;
  for (var i = 0; i < 5; i = i + 1)
  {
    var h = 0.001 + 0.15 * f32(i) / 4.0;
    var d = scene(current + h * normal).w;
    occ += (h - d) * sca;
    sca *= 0.95;
  }

  return clamp( 1.0 - 2.0 * occ, 0.0, 1.0 ) * (0.5 + 0.5 * normal.y);
}

fn get_ambient_light(light_pos: vec3f, sun_color: vec3f, rd: vec3f) -> vec3f
{
  var backgroundcolor1 = int_to_rgb(i32(uniforms[12]));
  var backgroundcolor2 = int_to_rgb(i32(uniforms[29]));
  var backgroundcolor3 = int_to_rgb(i32(uniforms[30]));
  
  var ambient = backgroundcolor1 - rd.y * rd.y * 0.5;
  ambient = mix(ambient, 0.85 * backgroundcolor2, pow(1.0 - max(rd.y, 0.0), 4.0));

  var sundot = clamp(dot(rd, normalize(vec3f(light_pos))), 0.0, 1.0);
  var sun = 0.25 * sun_color * pow(sundot, 5.0) + 0.25 * vec3f(1.0,0.8,0.6) * pow(sundot, 64.0) + 0.2 * vec3f(1.0,0.8,0.6) * pow(sundot, 512.0);
  ambient += sun;
  ambient = mix(ambient, 0.68 * backgroundcolor3, pow(1.0 - max(rd.y, 0.0), 16.0));

  return ambient;
}

fn get_light(current: vec3f, obj_color: vec3f, rd: vec3f) -> vec3f
{
  var light_p = vec3f(uniforms[13], uniforms[14], uniforms[15]);
  var sun_color = int_to_rgb(i32(uniforms[16]));
  var ambient = get_ambient_light(light_p, sun_color, rd);
  var normal = get_normal(current);

  // calculate light based on the normal
  // if the object is too far away from the light source, return ambient light
  if (length(current) > uniforms[20] + uniforms[8])
  {
    return ambient;
  }

  // calculate the light intensity
  // Use:

  // - shadow
  var light_dir = normalize(light_p - current);
  var intensit = max(dot(normal, light_dir), 0.0);
  var diffuse = intensit * obj_color * sun_color;
  var shadow = get_soft_shadow(current + normal * 0.001, light_dir, 0.001, length(light_dir - current), 32.0);

  var diffuse_light = diffuse * sun_color * shadow;
  // - ambient light
  var direct = normalize(-rd);
  var reflect_dir = reflect(-light_dir, normal);
  var specular = pow(max(dot(direct, reflect_dir), 0.0), 32.0);
  // - object color
  var ao = get_AO(current, normal);

  var final_light = ambient * obj_color + (diffuse_light + 0.8 * specular);
  final_light *= ao;
  return clamp(final_light, vec3f(0.0), vec3f(1.0));
}

fn set_camera(ro: vec3f, ta: vec3f, cr: f32) -> mat3x3<f32>
{
  var cw = normalize(ta - ro);
  var cp = vec3f(sin(cr), cos(cr), 0.0);
  var cu = normalize(cross(cw, cp));
  var cv = normalize(cross(cu, cw));
  return mat3x3<f32>(cu, cv, cw);
}

fn animate(val: vec3f, time_scale: f32, offset: f32) -> vec3f
{
  return val + vec3f(
      sin(time_scale * uniforms[0] + offset),
      cos(time_scale * uniforms[0] + offset),
      sin(time_scale * uniforms[0] + offset) * cos(time_scale * uniforms[0] + offset)
  );
}

@compute @workgroup_size(THREAD_COUNT, 1, 1)
fn preprocess(@builtin(global_invocation_id) id : vec3u)
{
  let obj_id = id.x;
  var time = uniforms[0];
  var spheresCount = i32(uniforms[2]);
  var boxesCount = i32(uniforms[3]);
  var torusCount = i32(uniforms[4]);
  var all_objects_count = spheresCount + boxesCount + torusCount;

  if (id.x >= u32(all_objects_count))
  {
    return;
  }

  let shape_info = shapesinfob[obj_id];
  let shape_index = i32(shape_info.y);
  let shape = shapesb[shape_index];
  let animated_rotation = shape.animate_rotation.xyz * sin(time* shape.animate_rotation.w);
  let quat_animated = quaternion_from_euler(animated_rotation + shape.rotation.xyz);
  shapesb[obj_id].quat = quat_animated;
}

@compute @workgroup_size(THREAD_COUNT, THREAD_COUNT, 1)
fn render(@builtin(global_invocation_id) id : vec3u)
{
  // unpack data
  var fragCoord = vec2f(f32(id.x), f32(id.y));
  var rez = vec2f(uniforms[1]);
  var time = uniforms[0];

  // camera setup
  var lookfrom = vec3(uniforms[6], uniforms[7], uniforms[8]);
  var lookat = vec3(uniforms[9], uniforms[10], uniforms[11]);
  var camera = set_camera(lookfrom, lookat, 0.0);
  var ro = lookfrom;

  // get ray direction
  var uv = (fragCoord - 0.5 * rez) / rez.y;
  uv.y = -uv.y;
  var rd = camera * normalize(vec3(uv, 1.0));

  // call march function and get the color/depth
  let m_out = march(ro, rd);
  let depth = m_out.depth;

  var color: vec3f;
  if (depth < MAX_DIST)
  {
      // Ray hit an object
      var p = ro + rd * depth;
      color = get_light(p, m_out.color, rd);
  }
  else
  {
      // Ray missed all objects, use background color
      var light_p= vec3f(uniforms[13], uniforms[14], uniforms[15]);
      var sun_color = int_to_rgb(i32(uniforms[16]));
      color = get_ambient_light(light_p, sun_color, rd);
  }

  // Display the result
  color = linear_to_gamma(color);
  fb[mapfb(id.xy, uniforms[1])] = vec4f(color, 1.0);
}