use crate::colors::wavelength_to_stimul;

pub fn spectrum_file_to_rgb(path: &str) -> [f32; 3] {
    let data = std::fs::read_to_string(path).unwrap();
    let mut sum = [0.0f32; 3];
    let mut ri_sum = 0.0;
    let mut lines = data.lines();
    while let Some(l) = lines.next() {
        match l {
            "<tr class='odd'>" | "<tr class='even'>" => {
                lines.next();
                let ion_n = lines.next().unwrap();
                let ion = ion_n
                    [(" <td class=\"lft1\"><b>".len())..(ion_n.len() - "</b>&nbsp;</td>".len())]
                    .to_string();
                lines.next();
                lines.next();
                let wl_n = lines.next().unwrap().replace("&nbsp;", "");
                let wl = wl_n
                    [(" <td class=\"fix\">          ".len())..(wl_n.len() - "</td>".len())]
                    .to_string()
                    .parse::<f32>()
                    .unwrap();
                lines.next();
                let ri_n = lines.next().unwrap().replace("&nbsp;", "");
                let ri = ri_n[(" <td class=\"cnt\">".len())
                    ..(ri_n
                        .find("<a ")
                        .unwrap_or_else(|| ri_n.len() - "</td>".len()))]
                    .to_string()
                    .parse::<f32>()
                    .unwrap_or_else(|_| 0.0);
                let ea_n = lines.next().unwrap().replace("&nbsp;", "");
                let ea_2 = ea_n[(" <td class=\"lft1\">".len())
                    ..(ea_n.len() - "</td>".len())]
                    .to_string();
                if ea_2.len()==0{
                    continue;
                }
                let mut s=ea_2.split("e+");
                let ea=s.next().unwrap().parse::<f32>().unwrap()*10.0f32.powi(s.next().unwrap().parse::<i32>().unwrap());
                lines.next();
                let el_n=lines.next().unwrap().replace("&nbsp;", "").replace(" ", "");
                let el=el_n[("<tdclass=\"fix\"><spanid=\"038003.000092\"class=\"en_span\"onclick=\"selectById('038003.000092')\"onmouseover=\"setMOn(this)\"onmouseout=\"setMOff(this)\">".len())..(el_n.len()-"</span></td>".len())].to_string().parse::<f32>().unwrap();
                lines.next();
                let eu_n=lines.next().unwrap().replace("&nbsp;", "").replace(" ", "");
                let eu=eu_n[("<tdclass=\"fix\"><spanid=\"038003.000092\"class=\"en_span\"onclick=\"selectById('038003.000092')\"onmouseover=\"setMOn(this)\"onmouseout=\"setMOff(this)\">".len())..(eu_n.len()-"</span></td>".len())].to_string().parse::<f32>().unwrap();
                if ion.ends_with(" I") && wl > 360.0 && wl < 850.0 {
                    let boltzmann = (-eu / (8.617333262e-5 * 2000.0)).exp();
                    println!("{boltzmann}");
                    println!("{eu}");
                    let weight = 1.0 * ea * boltzmann;
                    let st = wavelength_to_stimul(wl);
                    sum[0] += st[0] * weight;
                    sum[1] += st[1] * weight;
                    sum[2] += st[2] * weight;
                    ri_sum += weight;
                    println!("{weight}");
                }
            }
            _ => {}
        }
    }
    sum[0] /= ri_sum;
    sum[1] /= ri_sum;
    sum[2] /= ri_sum;
    sum
}
