use std::num::NonZeroU32;

use bytemuck::{Pod, Zeroable};
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    Adapter, Buffer, BufferDescriptor, BufferUsages, CommandBuffer, CommandEncoder,
    CommandEncoderDescriptor, Device, Extent3d, ImageCopyBuffer, ImageCopyTexture, ImageDataLayout,
    Maintain, MapMode, Origin3d, Queue, Texture, TextureAspect, TextureFormat, RenderPipeline,
};

pub struct TexturePulldown<'a> {
    pub inner: &'a Texture,
    pub format: TextureFormat,
    pub size: Extent3d,
    pub samples: u8,
}

pub struct TestingContext {
    pub adapter: Adapter,
    pub adapter_info: wgt::AdapterInfo,
    pub adapter_downlevel_capabilities: wgt::DownlevelCapabilities,
    pub device: Device,
    pub device_features: wgt::Features,
    pub device_limits: wgt::Limits,
    pub queue: Queue,
}

impl TestingContext {
    pub fn encoder(&self) -> CommandEncoder {
        self.device
            .create_command_encoder(&CommandEncoderDescriptor::default())
    }

    fn full_buffer_usages() -> BufferUsages {
        let mut usages = BufferUsages::all();
        usages -= BufferUsages::MAP_READ | BufferUsages::MAP_WRITE | BufferUsages::INDEX;
        usages
    }

    pub fn map_read_buffer(&self, size: u64) -> Buffer {
        self.device.create_buffer(&BufferDescriptor {
            label: None,
            size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        })
    }

    pub fn buffer_with_data<T: Zeroable + Pod>(&self, data: &[T]) -> Buffer {
        self.device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(data),
            usage: Self::full_buffer_usages(),
        })
    }

    pub fn index_buffer_with_data<T: Zeroable + Pod>(&self, data: &[T]) -> Buffer {
        self.device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(data),
            usage: BufferUsages::INDEX | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        })
    }

    pub fn buffer_with_nonzero_values(&self, bytes: usize) -> Buffer {
        self.buffer_with_data(&vec![0xdeadbeef_u64; (bytes + 7) / 8])
    }

    pub fn submit_and_pulldown<CmdBufIter>(
        &self,
        cmd_bufs: CmdBufIter,
        buffers: &[&Buffer],
        textures: &[TexturePulldown<'_>],
    ) -> Vec<Vec<u8>>
    where
        CmdBufIter: IntoIterator<Item = CommandBuffer>,
    {
        let mut mapped_buffers = Vec::with_capacity(buffers.len() + textures.len());
        let mut encoder = self.encoder();

        for &buf in buffers {
            let map_buf = self.map_read_buffer(buf.size());

            encoder.copy_buffer_to_buffer(buf, 0, &map_buf, 0, buf.size());
            mapped_buffers.push(map_buf);
        }

        for tex in textures {
            let desc = tex.format.describe();
            let row_stride = (tex.size.width / desc.block_dimensions.0 as u32) as u64
                * desc.block_size as u64
                * tex.samples as u64;
            let row_stride_rounded = (row_stride + 255) & !255;
            let single_layer_columns = (tex.size.height / desc.block_dimensions.1 as u32) as u64;
            let all_columns = single_layer_columns * tex.size.depth_or_array_layers as u64;

            let total_bytes = row_stride_rounded * all_columns;

            let map_buf = self.map_read_buffer(total_bytes);

            encoder.copy_texture_to_buffer(
                ImageCopyTexture {
                    texture: tex.inner,
                    mip_level: 0,
                    origin: Origin3d::ZERO,
                    aspect: TextureAspect::All,
                },
                ImageCopyBuffer {
                    buffer: &map_buf,
                    layout: ImageDataLayout {
                        offset: 0,
                        bytes_per_row: NonZeroU32::new(row_stride_rounded as u32),
                        rows_per_image: NonZeroU32::new(single_layer_columns as u32),
                    },
                },
                Extent3d {
                    width: tex.size.width * tex.samples as u32,
                    ..tex.size
                },
            )
        }

        self.queue
            .submit(cmd_bufs.into_iter().chain(Some(encoder.finish())));
        self.device.poll(Maintain::Wait);

        let mut output_data = Vec::with_capacity(mapped_buffers.len());
        for map_buf in &mapped_buffers {
            map_buf.slice(..).map_async(MapMode::Read, |_| ());

            output_data.push(map_buf.slice(..).get_mapped_range().to_vec())
        }

        output_data
    }
}

