using NCDatasets
using PyPlot
using Statistics
using DataFrames
using Printf


# analysis folder
#ana_folder = "/central/scratch/bischtob/heldsuarez/nc"
ana_folder = "/Users/lenka/Desktop/CLIMA/output"
ana_folder = "/groups/esm/lenka/CLIMA/output"
num_avg_a = 350
num_avg_b = num_avg_a + 12
num_avg = num_avg_b - num_avg_a + 1

# read the Netcdf files from disk
function read_ncfiles(filename, filename_aux, pinfo)
  path = joinpath(ana_folder, filename)
  dfile = Dataset(path)
  if pinfo==true
    print(dfile.attrib)
  end
  ρ = dfile["ro"].var[:]
  u = dfile["rou"].var[:] ./ ρ
  e = dfile["roe"].var[:] ./ ρ
  lat = dfile["lat"].var[:]
  lon = dfile["long"].var[:]
  rad = dfile["rad"].var[:] .- 6371000.0
  path = joinpath(ana_folder, filename_aux)
  dfile = Dataset(path)
  #T = dfile["moisture.air_T"].var[:] 
  return lon, lat, rad, ρ, u, e
end




# get variables
lon, lat, rad, ρ, u, e = read_ncfiles("hs_step0001.nc", "hs_step0001_aux.nc", true)

# load data into arrays for averaging
ρ_a = zeros(num_avg, size(rad)[1], size(lat)[1], size(lon)[1])
u_a = zeros(num_avg, size(rad)[1], size(lat)[1], size(lon)[1])
e_a = zeros(num_avg, size(rad)[1], size(lat)[1], size(lon)[1])
T_a = zeros(num_avg, size(rad)[1], size(lat)[1], size(lon)[1])
for i in 1:9
  ind = num_avg_a + i - 1
  lon, lat, rad, ρ_a[i,:,:,:], u_a[i,:,:,:], e_a[i,:,:,:] = read_ncfiles(@sprintf("hs_step000%s.nc",i), @sprintf("hs_step000%s_aux.nc",i), false)
end

# average data
u_zm = dropdims( mean(u_a, dims = (1,4) ) , dims = (1,4) )
T_zm = dropdims( mean(e_a, dims = (1,4) ) , dims = (1,4) )

# plot
contourf(lat*180/3.14 .- 90.0, rad, u_zm, levels=10)

xlabel!("latitude")
ylabel!("z [m]")
savefig("zonal_mean_u.png")

c=contourf(lat*180/3.14 .- 90.0, rad, T_zm, levels=10)

xlabel!("latitude")
ylabel!("z [m]")
savefig("zonal_mean_T.png")


