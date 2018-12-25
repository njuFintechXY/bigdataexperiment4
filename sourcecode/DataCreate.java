package skl;

import java.io.IOException;
import java.util.List;
import java.util.ArrayList;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.KeyValueTextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.util.LineReader;

import org.apdplat.word.WordSegmenter;
import org.apdplat.word.segmentation.Word;

//用于创建能用于分类的数据集
//将文本转化为多维空间中的点集
public class DataCreate{
	//map读取训练集文件，分词，计算特征向量
	public static class DCMapper extends Mapper<Text,Text,Text,Text>{
		private String[] tez;//特征词
		protected void setup(Context context) {
			tez = context.getConfiguration().getStrings("tez");
		}
		public void map(Text key,Text value,Context context) throws IOException,InterruptedException{
			int[] vec = new int[tez.length];
			List<Word> words = WordSegmenter.seg(key.toString());
			for(Word w:words) {
				String wo = w.getText();
				for(int i=0;i<tez.length;i++) {
					if(tez[i].equals(wo)) {
						vec[i]+=1;
						break;
					}
				}
			}
			String keyres = Integer.toString(vec[0]);
			for(int j = 1; j<tez.length;j++) {
				keyres += " "+Integer.toString(vec[j]);
			}
			Text keyresult = new Text();
			keyresult.set(keyres);
			context.write(keyresult, value);
		}
		
	}
	
	public static String[] gettez(Configuration conf, Path inputpath) throws IOException{
		//获取特征词
		ArrayList<String> tzc = new ArrayList<String>();
		Text line = new Text();
		FileSystem fs = FileSystem.get(conf);
		FSDataInputStream fsi = fs.open(inputpath);
		LineReader lr = new LineReader(fsi,conf);
		while(lr.readLine(line)>0) {
			tzc.add(line.toString());
		}
		lr.close();
		String[] b = (String[])tzc.toArray(new String[tzc.size()]);
		return b;
	}
	
	public static void main(String[] args) throws Exception{
		if(args.length != 3) {
			System.err.println("Usage: Datacreate <in> <ci.txt> <out>");
			System.exit(2);
		}
		Configuration conf = new Configuration();
		Path inputPath = new Path(args[0]);
		Path ciPath = new Path(args[1]);
		Path outputPath = new Path(args[2]);
		
		String[] tzc = gettez(conf,ciPath);
		//System.out.println(tzc[0]+"+"+tzc[2]);
		conf.setStrings("tez", tzc);
		FileSystem fs = FileSystem.get(conf);
		if(fs.exists(outputPath))
			fs.delete(outputPath,true);
		Job job = Job.getInstance(conf,"createceshijob");
		job.setJarByClass(DataCreate.class);
		job.setMapperClass(DCMapper.class);
		job.setNumReduceTasks(1);
		job.setInputFormatClass(KeyValueTextInputFormat.class);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);
		FileInputFormat.addInputPath(job, inputPath);
		FileOutputFormat.setOutputPath(job, outputPath);
		System.exit(job.waitForCompletion(true)?0:1);
	}
}




